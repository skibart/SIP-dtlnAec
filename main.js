import { Inviter, Registerer, SessionState, UserAgent, Web } from 'https://cdn.jsdelivr.net/npm/sip.js@0.21.2/lib/index.min.js'

const BLOCK_LEN = 512
const BLOCK_SHIFT = 128

class DtlnAec {
    constructor(interpreter1, interpreter2) {
        this.interpreter1 = interpreter1
        this.interpreter2 = interpreter2
        if (!interpreter1.inputs[1] || !interpreter1.inputs[1].shape) {
            throw Error('Failed DTLN AEC model 1.')
        }
        this.statesShape1 = interpreter1.inputs[1].shape

        if (!interpreter2.inputs[1] || !interpreter2.inputs[1].shape) {
            throw Error('Failed DTLN AEC model 2.')
        }
        this.statesShape2 = interpreter2.inputs[1].shape

        this.states1 = new Float32Array(this.statesShape1.reduce((a, b) => a * b))
        this.states2 = new Float32Array(this.statesShape2.reduce((a, b) => a * b))

        this.inBuffer = new Float32Array(BLOCK_LEN)
        this.inBufferLpb = new Float32Array(BLOCK_LEN)
        this.outBuffer = new Float32Array(BLOCK_LEN)

        this.inBufferEnd = BLOCK_LEN - BLOCK_SHIFT

        this.timestamp = 0
    }

    static async loadModel(assetsPath, options = {}) {
        if (!assetsPath.endsWith('/')) {
            assetsPath += '/'
        }
        let modelName = options.modelName
        if (modelName === undefined) {
            modelName = 'dtln_aec_128'
        }
        const interpreter1 = await tflite.loadTFLiteModel(assetsPath + modelName + '_1.tflite')
        const interpreter2 = await tflite.loadTFLiteModel(assetsPath + modelName + '_2.tflite')

        return new DtlnAec(interpreter1, interpreter2)
    }

    processOutputAudioData(data) {
        const buffer = this.getSamplesFromAudioData(data)
        this.inBufferLpb.copyWithin(0, buffer.length)
        this.inBufferLpb.set(buffer, BLOCK_LEN - buffer.length)
    }

    processInputAudioData(data) {
        const tmpBuffer = this.getSamplesFromAudioData(data)

        if (this.inBufferEnd === BLOCK_LEN - BLOCK_SHIFT) {
            this.timestamp = data.timestamp ?? 0
        }

        const processedAudioDataList = []
        let tmpBufferOffset = 0

        while (tmpBufferOffset < tmpBuffer.length) {
            const n = Math.min(BLOCK_LEN - this.inBufferEnd, tmpBuffer.length - tmpBufferOffset)

            this.inBuffer.set(tmpBuffer.subarray(tmpBufferOffset, tmpBufferOffset + n), this.inBufferEnd)
            tmpBufferOffset += n
            this.inBufferEnd += n

            if (this.inBufferEnd === BLOCK_LEN) {
                const processedData = this.processInBuffer()

                processedAudioDataList.push(
                    new AudioData({
                        format: data.format,
                        sampleRate: 16000,
                        numberOfFrames: BLOCK_SHIFT,
                        numberOfChannels: 1,
                        timestamp: this.timestamp,
                        data: processedData,
                    }),
                )

                // 8ms (128 próbek / 16kHz)
                this.timestamp += 8000
                this.inBuffer.copyWithin(0, BLOCK_SHIFT)
                this.inBufferEnd = BLOCK_LEN - BLOCK_SHIFT
            }
        }

        return processedAudioDataList
    }

    getSamplesFromAudioData(data) {
        if (data.numberOfChannels !== 1) {
            throw Error('Stereo not supported.')
        }
        let tmpBuffer = new Float32Array(data.numberOfFrames)
        data.copyTo(tmpBuffer, { planeIndex: 0 })

        if (data.sampleRate === 48000) {
            const tmpDownsampledBuffer = new Float32Array(data.numberOfFrames / 3)
            let writeIndex = 0
            for (let i = 0; i < tmpBuffer.length; i += 3) {
                tmpDownsampledBuffer[writeIndex++] = tmpBuffer[i]
            }
            tmpBuffer = tmpDownsampledBuffer
        } else if (data.sampleRate !== 16000) {
            throw Error(`sampleRate=${data.sampleRate}.`)
        }
        return tmpBuffer
    }

    processInBuffer() {
        const outBlock = tf.tidy(() => {
            const inBlockFft = tf.spectral.rfft(tf.tensor1d(this.inBuffer))
            const inMag = tf.reshape(tf.abs(inBlockFft), [1, 1, -1])

            const lpbBlockFft = tf.spectral.rfft(tf.tensor1d(this.inBufferLpb))
            const lpbMag = tf.reshape(tf.abs(lpbBlockFft), [1, 1, -1])

            const output1 = this.interpreter1.predict([inMag, tf.reshape(tf.tensor1d(this.states1), this.statesShape1), lpbMag])
            const outMask = output1.Identity
            this.states1 = Float32Array.from(output1.Identity_1.dataSync())

            const estimatedBlock = tf.spectral.irfft(tf.mul(inBlockFft, outMask))

            const inLpb = tf.reshape(tf.tensor1d(this.inBufferLpb), [1, 1, -1])
            const output2 = this.interpreter2.predict([
                estimatedBlock,
                tf.reshape(tf.tensor1d(this.states2), this.statesShape2),
                inLpb,
            ])
            const finalBlock = output2.Identity
            this.states2 = Float32Array.from(output2.Identity_1.dataSync())

            return finalBlock.dataSync()
        })

        this.outBuffer.copyWithin(0, BLOCK_SHIFT)
        this.outBuffer.subarray(BLOCK_LEN - BLOCK_SHIFT).fill(0)
        for (let i = 0; i < BLOCK_LEN; i++) {
            this.outBuffer[i] += outBlock[i]
        }
        return this.outBuffer.slice(0, BLOCK_SHIFT)
    }
}

async function prepareFilterAudioStream() {
    const dtlnAec = await DtlnAec.loadModel('./assets', {
        modelName: 'dtln_aec_128',
    })
    console.log('DTLN-AEC załadowany.')

    const constraints = {
        audio: {
            echoCancellation: false,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
        },
    }
    const micStream = await navigator.mediaDevices.getUserMedia(constraints)
    const micTrack = micStream.getAudioTracks()[0]

    const trackProcessor = new MediaStreamTrackProcessor({ track: micTrack })
    const trackGenerator = new MediaStreamTrackGenerator({ kind: 'audio' })

    const reader = trackProcessor.readable.getReader()
    const writer = trackGenerator.writable.getWriter()

    async function processLoop() {
        try {
            const { value: inFrame, done } = await reader.read()
            if (done) {
                return
            }
            const outFrames = dtlnAec.processInputAudioData(inFrame)
            for (const frame of outFrames) {
                await writer.write(frame)
            }
            processLoop()
        } catch (e) {
            console.error('Błąd processLoop:', e)
        }
    }
    processLoop()

    const filteredStream = new MediaStream()
    filteredStream.addTrack(trackGenerator)
    return filteredStream
}

async function handleRemoteStream(remoteStream) {
    const remoteTrack = remoteStream.getAudioTracks()[0]
    if (!remoteTrack) {
        return
    }
    const remoteProcessor = new MediaStreamTrackProcessor({ track: remoteTrack })
    const remoteReader = remoteProcessor.readable.getReader()

    async function remoteLoop() {
        try {
            while (true) {
                const { value: frame, done } = await remoteReader.read()
                if (done || !frame) {
                    break
                }
                dtlnAec.processOutputAudioData(frame)
            }
        } catch (err) {
            console.error('Remote loop error:', err)
        }
    }
    remoteLoop()
}

const myMediaStreamFactory = async (constraints, sessionDescriptionHandler) => {
    const filteredStream = await prepareFilterAudioStream()
    return Promise.resolve(filteredStream)
}

const mySessionDescriptionHandlerFactory = (session, options) => {
    const mediaStreamFactory = myMediaStreamFactory
    const iceGatheringTimeout = 500
    const sessionDescriptionHandlerConfiguration = {
        iceGatheringTimeout,
        peerConnectionConfiguration: {
            ...Web.defaultPeerConnectionConfiguration(),
        },
    }
    const logger = session.userAgent.getLogger('sip.SessionDescriptionHandler')
    return new Web.SessionDescriptionHandler(logger, mediaStreamFactory, sessionDescriptionHandlerConfiguration)
}

const userAgentOptions = {
    uri: UserAgent.makeURI('sip:'),
    transportOptions: {
        server: 'wss://',
    },
    authorizationPassword: 'pass',
    hackViaTcp: false,
    hackIpInContact: false,
    sessionDescriptionHandlerFactory: mySessionDescriptionHandlerFactory,
    sessionDescriptionHandlerFactoryOptions: {
        peerConnectionConfiguration: {
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
        },
        constraints: {
            audio: true,
            video: false,
        },
        iceGatheringTimeout: 500,
    },
}

const userAgent = new UserAgent(userAgentOptions)
let currentSession = null

userAgent.delegate = {
    onInvite(invitation) {
        const incomingSession = invitation
        currentSession = incomingSession

        incomingSession.stateChange.addListener((newState) => {
            switch (newState) {
                case SessionState.Establishing:
                    break
                case SessionState.Established:
                    break
                case SessionState.Terminated:
                    break
                default:
                    break
            }
        })

        const constraintsDefault = {
            audio: true,
            video: false,
        }

        const options = {
            sessionDescriptionHandlerOptions: {
                constraints: constraintsDefault,
            },
        }

        incomingSession.accept(options)
    },
}

const assignStream = async (stream) => {
    const element = document.getElementById('audio')
    element.autoplay = true
    element.srcObject = stream
    await element.play()
}

const startUserAgent = async () => {
    await userAgent.start()
    const registererOptions = {}
    const registerer = new Registerer(userAgent, registererOptions)
    await registerer.register()
}

const initiateCall = async (phoneNumber) => {
    const target = UserAgent.makeURI(`sip:${phoneNumber}@`)
    if (!target) {
        throw new Error('Failed to create target URI.')
    }
    const inviterOptions = {}
    const inviter = new Inviter(userAgent, target, inviterOptions)
    const outgoingSession = inviter
    currentSession = outgoingSession

    outgoingSession.stateChange.addListener((newState) => {
        switch (newState) {
            case SessionState.Establishing:
                const sessionDescriptionHandler = outgoingSession.sessionDescriptionHandler
                if (!sessionDescriptionHandler || !(sessionDescriptionHandler instanceof Web.SessionDescriptionHandler)) {
                    throw new Error('Invalid session description handler.')
                }
                assignStream(sessionDescriptionHandler.remoteMediaStream)
                handleRemoteStream(sessionDescriptionHandler.remoteMediaStream)
                break
            case SessionState.Established:
                break
            case SessionState.Terminated:
                break
            default:
                break
        }
    })

    await inviter.invite()
}

const terminateCall = async () => {
    if (currentSession) {
        await currentSession.bye()
        currentSession = null
    } else {
        console.log('No active session')
    }
}

window.addEventListener('DOMContentLoaded', () => {
    document.getElementById('startUserAgent').addEventListener('click', startUserAgent)
    document.getElementById('initiateCall').addEventListener('click', () => {
        const phoneNumber = document.getElementById('phoneNumber').value
        initiateCall(phoneNumber)
    })
    document.getElementById('terminateCall').addEventListener('click', terminateCall)
})

export { startUserAgent, initiateCall, terminateCall }
