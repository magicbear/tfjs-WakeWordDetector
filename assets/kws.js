var FFTJS=function(t){function r(e){if(i[e])return i[e].exports;var o=i[e]={i:e,l:!1,exports:{}};return t[e].call(o.exports,o,o.exports,r),o.l=!0,o.exports}var i={};return r.m=t,r.c=i,r.i=function(t){return t},r.d=function(t,i,e){r.o(t,i)||Object.defineProperty(t,i,{configurable:!1,enumerable:!0,get:e})},r.n=function(t){var i=t&&t.__esModule?function(){return t.default}:function(){return t};return r.d(i,"a",i),i},r.o=function(t,r){return Object.prototype.hasOwnProperty.call(t,r)},r.p="",r(r.s=0)}([function(t,r,i){"use strict";function e(t){if(this.size=0|t,this.size<=1||0!=(this.size&this.size-1))throw new Error("FFT size must be a power of two and bigger than 1");this._csize=t<<1;for(var r=new Array(2*this.size),i=0;i<r.length;i+=2){var e=Math.PI*i/this.size;r[i]=Math.cos(e),r[i+1]=-Math.sin(e)}this.table=r;for(var o=0,n=1;this.size>n;n<<=1)o++;this._width=o%2==0?o-1:o,this._bitrev=new Array(1<<this._width);for(var s=0;s<this._bitrev.length;s++){this._bitrev[s]=0;for(var a=0;a<this._width;a+=2){var h=this._width-a-2;this._bitrev[s]|=(s>>>a&3)<<h}}this._out=null,this._data=null,this._inv=0}t.exports=e,e.prototype.fromComplexArray=function(t,r){for(var i=r||new Array(t.length>>>1),e=0;e<t.length;e+=2)i[e>>>1]=t[e];return i},e.prototype.createComplexArray=function(){for(var t=new Array(this._csize),r=0;r<t.length;r++)t[r]=0;return t},e.prototype.toComplexArray=function(t,r){for(var i=r||this.createComplexArray(),e=0;e<i.length;e+=2)i[e]=t[e>>>1],i[e+1]=0;return i},e.prototype.completeSpectrum=function(t){for(var r=this._csize,i=r>>>1,e=2;e<i;e+=2)t[r-e]=t[e],t[r-e+1]=-t[e+1]},e.prototype.transform=function(t,r){if(t===r)throw new Error("Input and output buffers must be different");this._out=t,this._data=r,this._inv=0,this._transform4(),this._out=null,this._data=null},e.prototype.realTransform=function(t,r){if(t===r)throw new Error("Input and output buffers must be different");this._out=t,this._data=r,this._inv=0,this._realTransform4(),this._out=null,this._data=null},e.prototype.inverseTransform=function(t,r){if(t===r)throw new Error("Input and output buffers must be different");this._out=t,this._data=r,this._inv=1,this._transform4();for(var i=0;i<t.length;i++)t[i]/=this.size;this._out=null,this._data=null},e.prototype._transform4=function(){var t,r,i=this._out,e=this._csize,o=this._width,n=1<<o,s=e/n<<1,a=this._bitrev;if(4===s)for(t=0,r=0;t<e;t+=s,r++){var h=a[r];this._singleTransform2(t,h,n)}else for(t=0,r=0;t<e;t+=s,r++){var f=a[r];this._singleTransform4(t,f,n)}var u=this._inv?-1:1,_=this.table;for(n>>=2;n>=2;n>>=2){s=e/n<<1;var l=s>>>2;for(t=0;t<e;t+=s)for(var p=t+l,v=t,c=0;v<p;v+=2,c+=n){var d=v,m=d+l,y=m+l,b=y+l,w=i[d],g=i[d+1],z=i[m],T=i[m+1],x=i[y],A=i[y+1],C=i[b],E=i[b+1],F=w,I=g,M=_[c],R=u*_[c+1],O=z*M-T*R,P=z*R+T*M,j=_[2*c],S=u*_[2*c+1],J=x*j-A*S,k=x*S+A*j,q=_[3*c],B=u*_[3*c+1],D=C*q-E*B,G=C*B+E*q,H=F+J,K=I+k,L=F-J,N=I-k,Q=O+D,U=P+G,V=u*(O-D),W=u*(P-G),X=H+Q,Y=K+U,Z=H-Q,$=K-U,tt=L+W,rt=N-V,it=L-W,et=N+V;i[d]=X,i[d+1]=Y,i[m]=tt,i[m+1]=rt,i[y]=Z,i[y+1]=$,i[b]=it,i[b+1]=et}}},e.prototype._singleTransform2=function(t,r,i){var e=this._out,o=this._data,n=o[r],s=o[r+1],a=o[r+i],h=o[r+i+1],f=n+a,u=s+h,_=n-a,l=s-h;e[t]=f,e[t+1]=u,e[t+2]=_,e[t+3]=l},e.prototype._singleTransform4=function(t,r,i){var e=this._out,o=this._data,n=this._inv?-1:1,s=2*i,a=3*i,h=o[r],f=o[r+1],u=o[r+i],_=o[r+i+1],l=o[r+s],p=o[r+s+1],v=o[r+a],c=o[r+a+1],d=h+l,m=f+p,y=h-l,b=f-p,w=u+v,g=_+c,z=n*(u-v),T=n*(_-c),x=d+w,A=m+g,C=y+T,E=b-z,F=d-w,I=m-g,M=y-T,R=b+z;e[t]=x,e[t+1]=A,e[t+2]=C,e[t+3]=E,e[t+4]=F,e[t+5]=I,e[t+6]=M,e[t+7]=R},e.prototype._realTransform4=function(){var t,r,i=this._out,e=this._csize,o=this._width,n=1<<o,s=e/n<<1,a=this._bitrev;if(4===s)for(t=0,r=0;t<e;t+=s,r++){var h=a[r];this._singleRealTransform2(t,h>>>1,n>>>1)}else for(t=0,r=0;t<e;t+=s,r++){var f=a[r];this._singleRealTransform4(t,f>>>1,n>>>1)}var u=this._inv?-1:1,_=this.table;for(n>>=2;n>=2;n>>=2){s=e/n<<1;var l=s>>>1,p=l>>>1,v=p>>>1;for(t=0;t<e;t+=s)for(var c=0,d=0;c<=v;c+=2,d+=n){var m=t+c,y=m+p,b=y+p,w=b+p,g=i[m],z=i[m+1],T=i[y],x=i[y+1],A=i[b],C=i[b+1],E=i[w],F=i[w+1],I=g,M=z,R=_[d],O=u*_[d+1],P=T*R-x*O,j=T*O+x*R,S=_[2*d],J=u*_[2*d+1],k=A*S-C*J,q=A*J+C*S,B=_[3*d],D=u*_[3*d+1],G=E*B-F*D,H=E*D+F*B,K=I+k,L=M+q,N=I-k,Q=M-q,U=P+G,V=j+H,W=u*(P-G),X=u*(j-H),Y=K+U,Z=L+V,$=N+X,tt=Q-W;if(i[m]=Y,i[m+1]=Z,i[y]=$,i[y+1]=tt,0!==c){if(c!==v){var rt=N,it=-Q,et=K,ot=-L,nt=-u*X,st=-u*W,at=-u*V,ht=-u*U,ft=rt+nt,ut=it+st,_t=et+ht,lt=ot-at,pt=t+p-c,vt=t+l-c;i[pt]=ft,i[pt+1]=ut,i[vt]=_t,i[vt+1]=lt}}else{var ct=K-U,dt=L-V;i[b]=ct,i[b+1]=dt}}}},e.prototype._singleRealTransform2=function(t,r,i){var e=this._out,o=this._data,n=o[r],s=o[r+i],a=n+s,h=n-s;e[t]=a,e[t+1]=0,e[t+2]=h,e[t+3]=0},e.prototype._singleRealTransform4=function(t,r,i){var e=this._out,o=this._data,n=this._inv?-1:1,s=2*i,a=3*i,h=o[r],f=o[r+i],u=o[r+s],_=o[r+a],l=h+u,p=h-u,v=f+_,c=n*(f-_),d=l+v,m=p,y=-c,b=l-v,w=p,g=c;e[t]=d,e[t+1]=0,e[t+2]=m,e[t+3]=y,e[t+4]=b,e[t+5]=0,e[t+6]=w,e[t+7]=g}}]);

const SAMPLE_RATE = 16000

function hannWindow(length) {
    const win = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
    }
    return win;
}


function applyWindow(buffer, win) {
    if (buffer.length !== win.length) {
        console.error(
            `Buffer length ${buffer.length} != window length ${win.length}.`);
        return null;
    }
    const out = new Float32Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
        out[i] = win[i] * buffer[i];
    }
    return out;
}

function fft(y) {
    const fft = new FFTJS(y.length);
    const out = fft.createComplexArray();
    const data = fft.toComplexArray(y);
    fft.transform(out, data);
    return out;
}


// PyTorch风格零填充（center=True）
function padCenterWithZeros(data, padLen) {
    const padded = new Float32Array(data.length + 2 * padLen);
    padded.set(data, padLen); // 居中填充
    return padded;
}

// 精确分帧实现（与Librosa一致）
function frameWithPadding(signal, frameLength, hopLength) {
    const numFrames = Math.floor(1 + (signal.length - frameLength) / hopLength);
    return Array.from({ length: numFrames }, (_, i) => {
        const start = i * hopLength;
        const end = start + frameLength;
        return signal.slice(start, end); // 自动补零当长度不足
    });
}

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

class MelSpectrogram {
    constructor(params) {
        this.nFft = params.nFft || 2048;
        if (params.n_stft === undefined)
            params.n_stft = Math.floor(this.nFft / 2) + 1;
        this.melScale = new MelScale(params);
        this.winLength = params.winLength || nFft;
        this.hopLength = params.hopLength || Math.floor(winLength / 4);
        this.center = params.center !== undefined ? params.center : true;
        this.fftWindow = hannWindow(this.winLength);
    }

    forward(y, log_add) {
        // 修改3：替换为PyTorch风格的零填充
        y = padCenterWithZeros(y, Math.floor(this.nFft / 2));

        // 修改4：精确分帧实现
        const yFrames = frameWithPadding(y, this.nFft, this.hopLength);

        // 修改5：调整输出维度为nFft/2 +1
        const stftMatrix = new Array(Math.floor(this.nFft / 2) + 1)
            .fill().map(() => new Float32Array(yFrames.length));

        // 修改6：计算功率谱
        for (let i = 0; i < yFrames.length; i++) {
            const winBuffer = applyWindow(yFrames[i], this.fftWindow);
            const fftResult = fft(winBuffer);

            // 只保留有效频率并计算模平方
            for (let k = 0; k <= this.nFft / 2; k++) {
                const real = fftResult[k * 2];
                const imag = fftResult[k * 2 + 1];
                stftMatrix[k][i] = real * real + imag * imag;
            }
        }

        return this.melScale.forward(stftMatrix, log_add);
    }
}
class MelScale {
    constructor({
        n_mels = 128,
        sample_rate = 16000,
        f_min = 0.0,
        f_max = null,
        n_stft = 201,
        norm = null,
        mel_scale = 'htk'
    }) {
        // 参数验证
        this.n_mels = n_mels;
        this.sample_rate = sample_rate;
        this.f_min = f_min;
        this.f_max = f_max === null ? sample_rate / 2 : f_max;
        this.n_stft = n_stft;
        this.norm = norm;
        this.mel_scale = mel_scale;

        if (this.f_min > this.f_max) {
            throw new Error(`f_min (${f_min}) must be <= f_max (${this.f_max})`);
        }

        // 生成滤波器组
        this.fb = this.melscale_fbanks();
    }

    // 核心滤波器生成方法
    melscale_fbanks() {
        const { n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_scale } = this;

        // 初始化频率点
        const all_freqs = new Float32Array(n_stft);
        for (let i = 0; i < n_stft; i++) {
            all_freqs[i] = i * sample_rate / (2 * (n_stft - 1));
        }

        // 计算梅尔频率点
        const m_min = this.hz_to_mel(f_min);
        const m_max = this.hz_to_mel(f_max);
        const m_points = new Float32Array(n_mels + 2);
        for (let i = 0; i < n_mels + 2; i++) {
            m_points[i] = m_min + (m_max - m_min) * i / (n_mels + 1);
        }
        const hz_points = m_points.map(m => this.mel_to_hz(m));

        // 创建滤波器组
        const fb = Array.from({ length: n_mels }, () => new Float32Array(n_stft));

        for (let i = 0; i < n_mels; i++) {
            const left = hz_points[i];
            const center = hz_points[i + 1];
            const right = hz_points[i + 2];

            for (let j = 0; j < n_stft; j++) {
                const freq = all_freqs[j];

                if (freq < left || freq > right) continue;

                // 计算三角滤波器权重
                if (freq <= center) {
                    fb[i][j] = (freq - left) / (center - left);
                } else {
                    fb[i][j] = (right - freq) / (right - center);
                }

                // Slaney归一化
                if (norm === 'slaney') {
                    fb[i][j] *= 2.0 / (right - left);
                }
            }
        }

        return fb;
    }

    // 频率转换函数
    hz_to_mel(f) {
        if (this.mel_scale === 'htk') {
            return 2595 * Math.log10(1 + f / 700);
        }
        // Slaney公式
        const mel = f / (f + 700.0 / 2595.0);
        return 1127 * Math.log(1 + mel);
    }

    mel_to_hz(m) {
        if (this.mel_scale === 'htk') {
            return 700 * (Math.pow(10, m / 2595) - 1);
        }
        // Slaney逆变换
        return 700.0 * (Math.exp(m / 1127) - 1) / 2595.0;
    }

    // 前向计算
    forward(specgram, add_log) {
        /* 输入维度检查
        specgram: [..., freq, time]
        输出: [..., n_mels, time]
        */

        // 矩阵乘法实现
        const [freqDim, timeDim] = [specgram.length, specgram[0].length];
        const result = Array.from({ length: this.n_mels }, () =>
            new Float32Array(timeDim).fill(0));

        // 执行矩阵乘法: (freq, time) x (freq, mels) -> (mels, time)
        for (let m = 0; m < this.n_mels; m++) {
            for (let t = 0; t < timeDim; t++) {
                let sum = 0;
                for (let f = 0; f < freqDim; f++) {
                    sum += specgram[f][t] * this.fb[m][f];
                }
                if (add_log)
                {
                    result[m][t] = Math.log(sum + add_log);
                } else {
                    result[m][t] = sum;
                }
            }
        }

        return result;
    }
}


class TfMelScale {
    constructor(params) {
        this.n_mels = params.n_mels || 80;
        this.sample_rate = params.sample_rate || 16000;
        this.f_min = params.f_min || 0;
        this.f_max = params.f_max || this.sample_rate / 2;
        this.n_stft = params.n_stft || 257;
        this.norm = params.norm || null;
        this.mel_scale = params.mel_scale || 'htk';

        // 预计算滤波器组
        this.fb = tf.tidy(() => this.createMelFilterbanks().transpose());
    }

    createMelFilterbanks() {
        return tf.tidy(() => {
            // 生成频率分箱
            const allFreqs = tf.linspace(0, this.sample_rate / 2, this.n_stft);

            // 计算梅尔频率点
            const melMin = this.hzToMel(this.f_min);
            const melMax = this.hzToMel(this.f_max);
            const mPoints = tf.linspace(melMin, melMax, this.n_mels + 2);
            const hzPoints = mPoints.arraySync().map(m => this.melToHz(m));

            // 构建滤波器组张量
            return tf.stack(hzPoints.slice(0, -2).map((_, i) => {
                const [left, center, right] = hzPoints.slice(i, i + 3);
                const lower = tf.sub(allFreqs, left);
                const upper = tf.sub(right, allFreqs);

                const leftTriangle = tf.div(lower, center - left);
                const rightTriangle = tf.div(upper, right - center);

                const weights = tf.minimum(leftTriangle, rightTriangle)
                    .maximum(0)
                    .minimum(1.0);

                // Slaney归一化
                return this.norm === 'slaney' ?
                    weights.mul(tf.scalar(2.0 / (right - left))) :
                    weights;
            }), 0);
        });
    }

    hzToMel(f) {
        if (this.mel_scale === 'htk') {
            return 2595 * Math.log10(1 + f / 700);
        }
        return 1127 * Math.log(1 + f / (700 / 2595));
    }

    melToHz(m) {
        if (this.mel_scale === 'htk') {
            return 700 * (Math.pow(10, m / 2595) - 1);
        }
        return 700 / 2595 * (Math.exp(m / 1127) - 1);
    }

    async forward(specgram) {
        return await tf.tidy(() => {
            // 输入形状转换 [freq, time] => [time, freq]
            const input = tf.transpose(specgram);

            // 执行矩阵乘法：[time, freq] x [freq, mels] => [time, mels]
            const melSpec = tf.matMul(input, this.fb);

            // 转换回 [mels, time]
            return tf.transpose(melSpec).add(1e-7).log();
        });
    }
}


class TfMelSpectrogram {
    constructor (params = {}) {
        this.nFft = params.nFft || 2048;
        if (params.n_stft === undefined)
            params.n_stft = Math.floor(this.nFft / 2) + 1;
        this.melScale = new TfMelScale(params);
        this.winLength = params.winLength || nFft;
        this.hopLength = params.hopLength || Math.floor(winLength / 4);
        this.center = params.center !== undefined ? params.center : true;

        // 生成汉宁窗 (自动对齐nFft长度)
        this.win = tf.signal.hannWindow(this.winLength).pad([[0, this.nFft - this.winLength]]);

    }

    async forward(specgram) {
        // 参数处理
        const spec = await tf.tidy(() => {
            // 中心填充处理
            let yTensor = tf.tensor1d(specgram);
            if (this.center) {
                const padLen = Math.floor(this.nFft / 2);
                yTensor = tf.pad(yTensor, [[padLen, padLen]]);
            }

            // 2. 准确分帧计算
            const numFrames = Math.floor(1 + (yTensor.size - this.nFft) / this.hopLength);
            const indices = tf.add(
                tf.range(0, numFrames * this.hopLength, this.hopLength, 'int32').reshape([-1, 1]),
                tf.range(0, this.nFft, 1, 'int32')
            );

            // 3. 收集分帧数据 [numFrames, nFft]
            const frames = yTensor.gather(indices.reshape([-1])).reshape([numFrames, this.nFft]);

            // 4. 后续处理
            const windowed = frames.mul(this.win);
            const spec = tf.abs(tf.spectral.rfft(windowed)).square();

            yTensor.dispose();
            
            return spec.transpose(); // [freq, time]
        });

        const melSpec = await this.melScale.forward(spec);
        return melSpec;
        // const melSpecData = await melSpec;
        // return melSpecData;
        // add log
        // return await tf.tensor(melSpecData).add(1e-7).log();
    }
}


function getMonoAudio(audioBuffer) {
    if (audioBuffer.numberOfChannels === 1) {
        return audioBuffer.getChannelData(0);
    }
    if (audioBuffer.numberOfChannels !== 2) {
        throw Error(
            `${audioBuffer.numberOfChannels} channel audio is not supported.`);
    }
    const ch0 = audioBuffer.getChannelData(0);
    const ch1 = audioBuffer.getChannelData(1);

    const mono = new Float32Array(audioBuffer.length);
    for (let i = 0; i < audioBuffer.length; ++i) {
        mono[i] = (ch0[i] + ch1[i]) / 2;
    }
    return mono;
}

async function resampleAndMakeMono(audioBuffer, targetSr = SAMPLE_RATE) {
    if (audioBuffer.sampleRate === targetSr) {
        return getMonoAudio(audioBuffer);
    }
    const sourceSr = audioBuffer.sampleRate;
    const lengthRes = (audioBuffer.length * targetSr) / sourceSr;
    if (!isSafari) {
        const _offlineCtx = new OfflineAudioContext(
            audioBuffer.numberOfChannels, audioBuffer.duration * targetSr,
            targetSr);
        const bufferSource = _offlineCtx.createBufferSource();
        bufferSource.buffer = audioBuffer;
        bufferSource.connect(_offlineCtx.destination);
        bufferSource.start();
        return _offlineCtx.startRendering().then(
            (buffer) => buffer.getChannelData(0));
    } else {
        // Safari does not support resampling with WebAudio.
        console.log(
            'Safari does not support WebAudio resampling, so this may be slow.',
            'O&F', logging.Level.WARN);

        const originalAudio = getMonoAudio(audioBuffer);
        const resampledAudio = new Float32Array(lengthRes);
        resample(
            ndarray(resampledAudio, [lengthRes]),
            ndarray(originalAudio, [originalAudio.length]));
        return resampledAudio;
    }
}