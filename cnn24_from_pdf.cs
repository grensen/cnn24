// https://github.com/grensen/cnn24
// .NET 8
// copy code
// then press to close functions: (ctrl + m + o)
// use Release mode, Debug is slow!

using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

// 0. load MNIST data
Console.WriteLine("\nBegin convolutional neural network demo\n");
AutoData d = new(@"C:\cnn2024\", AutoData.Dataset.MNIST); // get data

// 1. init cnn + nn + hyperparameters
int[] cnn = { 1, 8, 24 }; // input layer: non-RGB = 1, RGB = 3
int[] filter = { 6, 6 }; // x * y dim for kernel
int[] stride = { 1, 3 }; // pooling with higher strides than 1
int[] net = { 784, 300, 300, 10 }; // nn
var LR = 0.005f;
var MOMENTUM = 0.5f;
var DROPOUT = 0.5f;
var SEED = 274024;
var FACTOR = 0.95f;
int BATCH = 42;
int EPOCHS = 50;

// 2.0 convolution dimensions
int[] dim = CnnDimensions(cnn.Length - 1, 28, filter, stride);
// 2.1 convolution steps for layerwise preparation
int[] cStep = CnnSteps(cnn, dim);
// 2.2 kernel steps for layerwise preparation
int[] kStep = KernelSteps(cnn, filter);
// 2.3 init visual based kernel weights
float[] kernel = InitConvKernel(cnn, filter, SEED);
// 2.4 init neural network weights w.r.t. cnn
float[] weights = InitNeuralNetWeights(net, cStep, SEED);

NetInfo();
Random rng = new(SEED); 
Stopwatch sw = Stopwatch.StartNew();
int[] indices = Enumerable.Range(0, 60000).ToArray(), 
    done = new int[60000];

// 4.0
Console.WriteLine("\nStarting training");
float[] delta = new float[weights.Length], kDelta = new float[kernel.Length];
for (int epoch = 0; epoch < EPOCHS; epoch++, LR *= FACTOR, MOMENTUM *= FACTOR)
    CnnTraining(true, indices, d.samplesTrainingF, d.labelsTraining, rng.Next(),
    done, cnn, filter, stride, kernel, kDelta, dim, cStep, kStep, net,
    weights, delta, 60000, epoch, BATCH, LR, MOMENTUM, DROPOUT);
Console.WriteLine($"Done after {(sw.Elapsed.TotalMilliseconds / 1000.0):F2}s\n");

// 5.0
CnnTesting(d.samplesTestF, d.labelsTest, cnn, filter, stride, kernel, dim, cStep, kStep, net,
weights, 10000); 
Console.WriteLine("\nEnd CNN demo");

//+-------------------------------------------------------------------------------------------+

// 2.0 conv dimensions
static int[] CnnDimensions(int cnn_layerLen, int startDimension, int[] filter, int[] stride)
{
    int[] dim = new int[cnn_layerLen + 1];
    for (int i = 0, c_dim = (dim[0] = startDimension); i < cnn_layerLen; i++)
        dim[i + 1] = c_dim = (c_dim - (filter[i] - 1)) / stride[i];
    return dim;
}
// 2.1 convolution steps
static int[] CnnSteps(int[] cnn, int[] dim)
{
    int[] cs = new int[cnn.Length + 1];
    cs[1] = dim[0] * dim[0]; // startDimension^2
    for (int i = 0, sum = cs[1]; i < cnn.Length - 1; i++)
        cs[i + 2] = sum += cnn[i + 1] * dim[i + 1] * dim[i + 1];
    return cs;
}
// 2.2 kernel steps in structure for kernel weights
static int[] KernelSteps(int[] cnn, int[] filter)
{
    int[] ks = new int[cnn.Length - 1];
    for (int i = 0; i < cnn.Length - 2; i++)
        ks[i + 1] += cnn[i + 0] * cnn[i + 1] * filter[i] * filter[i];
    return ks;
}
// 2.3 init kernel weights
static float[] InitConvKernel(int[] cnn, int[] filter, int seed)
{
    int cnn_weightLen = 0;
    for (int i = 0; i < cnn.Length - 1; i++)
        cnn_weightLen += cnn[i] * cnn[i + 1] * filter[i] * filter[i];
    float[] kernel = new float[cnn_weightLen]; Random rnd = new(seed);
    for (int i = 0, c = 0; i < cnn.Length - 1; i++) // each cnn layer
        for (int l = 0, f = filter[i]; l < cnn[i]; l++) // each input map
            for (int r = 0; r < cnn[i + 1]; r++) // each output map
                for (int col = 0; col < f; col++) // kernel y
                    for (int row = 0; row < f; row++, c++) // kernel x
                        kernel[c] = (rnd.NextSingle() * 2 - 1) /
                        MathF.Sqrt(cnn[i] * cnn[i + 1] * 0.5f);
    return kernel;
}
// 2.4 init neural network weights for cnn
static float[] InitNeuralNetWeights(int[] net, int[] convMapsStep, int seed)
{
    // 3.1.1 fit cnn output to nn input
    net[0] = convMapsStep[^1] - convMapsStep[^2];
    // 3.1.2 glorot nn weights init
    int len = 0;
    for (int n = 0; n < net.Length - 1; n++)
        len += net[n] * net[n + 1];
    float[] weight = new float[len];
    Random rnd = new(seed);
    for (int i = 0, m = 0; i < net.Length - 1; i++, m += net[i] * net[i - 1]) // layer
        for (int w = m; w < m + net[i] * net[i + 1]; w++) // weights
            weight[w] = (rnd.NextSingle() * 2 - 1)
            * MathF.Sqrt(6.0f / (net[i] + net[i + 1])) * 0.5f;
    return weight;
}
// 2.5 cnn number of neurons
static int CnnNeuronsLen(int startDimension, int[] cnn, int[] dim)
{
    int cnn_layerLen = cnn.Length - 1;
    int cnn_neuronLen = startDimension * startDimension; // add input first
    for (int i = 0; i < cnn_layerLen; i++)
        cnn_neuronLen += cnn[i + 1] * dim[i + 1] * dim[i + 1];
    return cnn_neuronLen;
}
// 2.6 nn number of neurons
static int NeuronsLen(int[] net)
{
    int sum = 0;
    for (int n = 0; n < net.Length; n++)
        sum += net[n];
    return sum;
}
// 3.0 cnn ff
static void ConvForward(int[] cnn, int[] dim, int[] cs, int[] filter,
int[] kStep, int[] stride, Span<float> conv, Span<float> kernel)
{
    for (int i = 0; i < cnn.Length - 1; i++)
    {
        int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1],
        lStep = cs[i + 0], rStep = cs[i + 1], kd = filter[i], ks = kStep[i], st = stride[i],
        lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd, sDim = st * lDim;
        // better cache locality?
        for (int l = 0, leftStep = lStep; l < left; l++, leftStep += lMap) // input map
        {
            var inpMap = conv.Slice(leftStep, lDim * lDim);
            for (int r = 0, rightStep = rStep; r < right; r++, rightStep += rMap) // out map
            {
                var outMap = conv.Slice(rightStep, rDim * rDim);
                for (int col = 0; col < kd; col++) // kernel filter dim y
                {
                    var kernelRow = kernel.Slice(ks + (l * right + r) * kMap + kd * col, kd);
                    for (int y = 0, kk = 0; y < rDim; y++) // conv dim y
                    {
                        var inputRow = inpMap.Slice((y * st + col) * lDim, lDim);
                        for (int x = 0; x < rDim; x++, kk++) // conv dim x
                        {
                            float sum = 0; // kernel filter dim x
                            for (int row = 0, rowStep = x * st; row < kernelRow.Length; row++)
                                sum = kernelRow[row] * inputRow[rowStep + row] + sum;
                            outMap[x + y * rDim] += sum;
                        }
                    }
                }
            }
        }
        // relu activation
        var activateMaps = conv.Slice(rStep, rDim * rDim * right);
        var vec = MemoryMarshal.Cast<float, Vector512<float>>(activateMaps);
        for (int v = 0; v < vec.Length; v++) // SIMD
            vec[v] = Vector512.Max(vec[v], Vector512<float>.Zero);
        for (int k = vec.Length * Vector512<float>.Count; k < activateMaps.Length; k++) // cm
            activateMaps[k] = MathF.Max(activateMaps[k], 0);
    }
}
// 3.1 dropout between cnn output and nn input
static void Dropout(int seed, Span<float> conv, int start, int part, float drop)
{
    Random rng = new(seed);
    var outMap = conv.Slice(start, part);
    for (int k = 0; k < outMap.Length; k++) // conv map
    {
        float sum = outMap[k];
        if (sum > 0)
            outMap[k] = rng.NextSingle() > drop ? sum : 0;
    }
}
// 3.2 nn ff
static void FeedForward(Span<int> net, Span<float> weights, Span<float> neurons)
{
    for (int i = 0, k = net[0], w = 0; i < net.Length - 1; i++) // layers
    {
        // slice input + output layer
        var outLocal = neurons.Slice(k, net[i + 1]);
        var inpLocal = neurons.Slice(k - net[i], net[i]);
        for (int l = 0; l < inpLocal.Length; l++, w = outLocal.Length + w) // input neurons
        {
            // fast input neuron
            var inpNeuron = inpLocal[l];
            if (inpNeuron <= 0) continue; // ReLU input pre-activation
                                          // slice connected weights
            var wts = weights.Slice(w, outLocal.Length);
            // span to vector
            var wtsVec = MemoryMarshal.Cast<float, Vector512<float>>(wts);
            var outVec = MemoryMarshal.Cast<float, Vector512<float>>(outLocal);
            // SIMD
            for (int v = 0; v < outVec.Length; v++)
                outVec[v] = wtsVec[v] * inpNeuron + outVec[v];
            // compute remaining output neurons
            for (int r = wtsVec.Length * Vector512<float>.Count; r < outLocal.Length; r++)
                outLocal[r] = wts[r] * inpNeuron + outLocal[r];
        }
        k = outLocal.Length + k; // stack output id
    }
}
// 3.3 softmax
static int SoftArgMax(Span<float> neurons)
{
    int id = 0; // argmax
    float max = neurons[0];
    for (int i = 1; i < neurons.Length; i++)
        if (neurons[i] > max) { max = neurons[i]; id = i; }
    // softmax activation
    float scale = 0;
    for (int n = 0; n < neurons.Length; n++)
        scale += neurons[n] = MathF.Exp((neurons[n] - max));
    for (int n = 0; n < neurons.Length; n++)
        neurons[n] /= scale; // pseudo probabilities
    return id; // return nn prediction
}
// 3.4 output error gradient (target - output)
static void ErrorGradient(Span<float> neurons, int target)
{
    for (int i = 0; i < neurons.Length; i++)
        neurons[i] = target == i ? 1 - neurons[i] : -neurons[i];
}
// 3.5 nn bp
static void Backprop(Span<float> neurons, Span<int> net, Span<float> weights,
Span<float> deltas)
{
    int j = neurons.Length - net[^1], k = neurons.Length, m = weights.Length;
    for (int i = net.Length - 2; i >= 0; i--)
    {
        int right = net[i + 1], left = net[i];
        k -= right; j -= left; m -= right * left;
        // slice input + output layer
        var inputNeurons = neurons.Slice(j, left);
        var outputGradients = neurons.Slice(k, right);
        for (int l = 0, w = m; l < left; l++, w += right)
        {
            var n = inputNeurons[l];
            if (n <= 0) { inputNeurons[l] = 0; continue; }
            // slice connected weights + deltas
            var wts = weights.Slice(w, right); // var inVec = Vector256.Create(n);
            var dts = deltas.Slice(w, right);
            // turn to vector
            var wtsVec = MemoryMarshal.Cast<float, Vector256<float>>(wts);
            var dtsVec = MemoryMarshal.Cast<float, Vector256<float>>(dts);
            var graVec = MemoryMarshal.Cast<float, Vector256<float>>(outputGradients);
            var sumVec = Vector256<float>.Zero;
            for (int v = 0; v < graVec.Length; v++) // SIMD, gradient sum and delta
            {
                var outGraVec = graVec[v];
                sumVec = wtsVec[v] * outGraVec + sumVec;
                dtsVec[v] = n * outGraVec + dtsVec[v];
            }
            // turn vector sum to float
            var sum = Vector256.Sum(sumVec);
            // compute remaining elements
            for (int r = graVec.Length * Vector256<float>.Count; r < wts.Length; r++)
            {
                var outGraSpan = outputGradients[r];
                sum = wts[r] * outGraSpan + sum;
                dts[r] = n * outGraSpan + dts[r];
            }
            inputNeurons[l] = sum; // reuse for gradients now
        }
    }
}
// 3.6 cnn bp
static void ConvBackprop(int[] cnn, int[] dim, int[] cSteps, int[] filter, int[] kStep, int[]
stride, Span<float> kernel, Span<float> kDelta, Span<float> cnnGradient, Span<float> conv)
{
    for (int i = cnn.Length - 2; i >= 0; i--) // one loop bp: cnn gradient and kernel delta
    {
        int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1],
        lStep = cSteps[i], rStep = cSteps[i + 1], kd = filter[i], ks = kStep[i], st =
        stride[i], lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd, sDim = st * lDim;
        for (int l = 0; l < left; l++, lStep += lMap) // input channel map
        {
            var inpMap = conv.Slice(lStep, lMap);
            var inpGraMap = cnnGradient.Slice(lStep, lMap);
            for (int r = 0, rs = rStep; r < right; r++, rs += rMap) // output channel map
            {
                var outMap = conv.Slice(rs, rMap);
                var graMap = cnnGradient.Slice(rs, rMap);
                for (int col = 0; col < kd; col++) // filter dim y cols
                {
                    int kernelID = ks + (l * right + r) * kMap + kd * col;
                    var kernelRow = kernel.Slice(kernelID, kd);
                    var kernelDeltaRow = kDelta.Slice(kernelID, kd);
                    for (int y = 0; y < rDim; y++) // conv dim y
                    {
                        int irStep = (y * st + col) * lDim;
                        var inputRow = inpMap.Slice(irStep, lDim);
                        var inputGraRow = inpGraMap.Slice(irStep, lDim);
                        int outStep = y * rDim;
                        for (int x = 0; x < rDim; x++) // conv dim x
                            if (outMap[x + outStep] > 0) // relu derivative
                            {
                                float gra = graMap[x + outStep];
                                for (int row = 0, rowStep = x * st; row < kd; row++) // fdx rw
                                {
                                    kernelDeltaRow[row] += inputRow[rowStep + row] * gra;
                                    inputGraRow[rowStep + row] += kernelRow[row] * gra;
                                }
                            }
                    }
                }
            }
        }
    }
}
// 3.7 sgd
static void Update(Span<float> weights, Span<float> delta, float lr, float mom)
{
    var weightVecArray = MemoryMarshal.Cast<float, Vector512<float>>(weights);
    var deltaVecArray = MemoryMarshal.Cast<float, Vector512<float>>(delta);
    // SIMD
    for (int v = 0; v < weightVecArray.Length; v++)
    {
        weightVecArray[v] = deltaVecArray[v] * lr + weightVecArray[v];
        deltaVecArray[v] *= mom;
    }
    // remaining elements
    for (int w = weightVecArray.Length * Vector512<float>.Count; w < weights.Length; w++)
    {
        weights[w] = delta[w] * lr + weights[w];
        delta[w] *= mom;
    }
}

// 4.0 train sample
static int TrainSample(int id, int target, int seedD, int[] done, float[] data,
int[] cnn, int[] dim, int[] filter, int[] stride, float[] kernel, float[] kernelDelta,
int[] cSteps, int[] kStep, int[] net, float[] weight, float[] delta,
int cnnLen, int nnLen, float drop)
{
    if (done[id] >= 5) return -1; // drop easy examples
                                  // feed conv input layer with sample
    Span<float> conv = new float[cnnLen];
    data.AsSpan().Slice(id * 784, 784).CopyTo(conv);
    // conv feed forward + dropout (cnn output == nn input) layer
    ConvForward(cnn, dim, cSteps, filter, kStep, stride, conv, kernel);
    int right = cnn[^1], rDim = dim[^1], start = cSteps[^2];
    Dropout(seedD, conv, start, rDim * rDim * right, drop);
    // neural net feed forward
    Span<float> neuron = new float[nnLen];
    conv.Slice(cSteps[^2], net[0]).CopyTo(neuron);
    FeedForward(net, weight, neuron);
    var outs = neuron.Slice(nnLen - net[^1], net[^1]);
    int prediction = SoftArgMax(outs);
    // drop easy examples
    if (outs[target] >= 0.9999) { done[id] += 1; return -1; }
    if (outs[target] >= 0.9) return prediction; // probability check
                                                // neural net backprop
    ErrorGradient(outs, target);
    Backprop(neuron, net, weight, delta);
    // conv net backprop
    Span<float> cnnGradient = new float[cSteps[^2] + net[0]];
    neuron.Slice(0, net[0]).CopyTo(cnnGradient.Slice(cSteps[^2], net[0]));
    ConvBackprop(cnn, dim, cSteps, filter, kStep,
    stride, kernel, kernelDelta, cnnGradient, conv);
    return prediction;
}

// 5.0 train epoch
static float CnnTraining(bool multiCPU, int[] indices, float[] data, byte[] label,
int seed, int[] done, int[] cnn, int[] filter, int[] stride, float[] kernel,
float[] kernelDelta, int[] dim, int[] cSteps, int[] kStep, int[] net, float[] weight,
float[] delta, int len, int epoch, int batch, float lr, float mom, float drop)
{
    DateTime elapsed = DateTime.Now; Random.Shared.Shuffle(indices); // shuffle ids
    int correct = 0, all = 0; Random rng = new Random(seed + epoch);
    int cnnLen = CnnNeuronsLen(28, cnn, dim), nnLen = NeuronsLen(net);
    // batch training
    for (int b = 0, B = (int)(len / batch); b < B; b++) // each batch for one epoch
    {
        int[] rand = Enumerable.Repeat(0, batch).Select(_ => rng.Next()).ToArray();
        if (multiCPU) Parallel.For(0, batch, x => // each sample in this batch
        {
            // get shuffled supervised sample id
            int id = indices[x + b * batch], target = label[id];
            int prediction = TrainSample(id, target, rand[x], done, data,
            cnn, dim, filter, stride, kernel, kernelDelta, cSteps,
            kStep, net, weight, delta, cnnLen, nnLen, drop);
            if (prediction != -1)
            {
                if (prediction == target) Interlocked.Increment(ref correct);
                Interlocked.Increment(ref all); // statistics accuracy
            }
        });
        else for (int x = 0; x < batch; x++)
            {
                int id = indices[x + b * batch], target = label[id];
                int prediction = TrainSample(id, target, rand[x], done, data,
                cnn, dim, filter, stride, kernel, kernelDelta, cSteps,
                kStep, net, weight, delta, cnnLen, nnLen, drop);
                if (prediction != -1)
                {
                    if (prediction == target) Interlocked.Increment(ref correct);
                    Interlocked.Increment(ref all); // statistics accuracy
                }
            }
        Update(kernel, kernelDelta, lr, 0); // no cnn mom works better?
        Update(weight, delta, lr, mom);
    }
    if ((epoch + 1) % 10 == 0)
        Console.WriteLine($"epoch = {(epoch + 1):00} | acc = {(correct * 100.0 / all):F2}%"
        + $" | time = {(DateTime.Now - elapsed).TotalSeconds:F2}s");
    return (correct * 100.0f / all);
}

// 6.0 test cnn on unseen data
static void CnnTesting(float[] data, byte[] label, int[] cnn, int[] filter, int[] stride,
float[] kernel, int[] dim, int[] cSteps, int[] kStep, int[] net, float[] weight, int len)
{
    DateTime elapsed = DateTime.Now;
    // cnn stuff
    int cnnLen = CnnNeuronsLen(28, cnn, dim), nnLen = NeuronsLen(net);
    // correction value for each neural network weight
    int correct = 0;
    Parallel.For(0, len, id =>
    {
        int target = label[id];
        Span<float> conv = new float[cnnLen];
        data.AsSpan().Slice(id * 784, 784).CopyTo(conv);
        // convolution feed forward
        ConvForward(cnn, dim, cSteps, filter, kStep, stride, conv, kernel);
        // copy cnn output to nn input
        Span<float> neuron = new float[nnLen];
        conv.Slice(cSteps[^2], net[0]).CopyTo(neuron);
        // neural net feed forward
        FeedForward(net, weight, neuron);
        int prediction = SoftArgMax(neuron.Slice(nnLen - net[^1], net[^1]));
        // statistics accuracy
        if (prediction == target)
            Interlocked.Increment(ref correct);
    });
    Console.WriteLine($"Test accuracy = {(correct * 100.0 / len):F2}%" +
    $" after {(DateTime.Now - elapsed).TotalSeconds:F2}s");
}

void NetInfo()
{
    #if DEBUG
    Console.WriteLine("Debug mode is on, switch to Release mode"); return;
    #endif

    Console.WriteLine($"Convolution = {string.Join("-", cnn)}");
    Console.WriteLine($"Kernel size = {string.Join("-", filter)}");
    Console.WriteLine($"Stride step = {string.Join("-", stride)}");
    Console.WriteLine($"DimensionX{" ",2}= {string.Join("-", dim)}");
    Console.WriteLine($"Map (Dim²){" ",2}= {string.Join("-", dim.Select(x => x * x))}");
    Console.WriteLine($"CNN+NN{" ",6}=" +
    $" {string.Join("-", cnn.Zip(dim, (x, d) => x * d * d))}+{string.Join("-", net)}");
    Console.WriteLine($"CNN weights{" ",1}= {kernel.Length} ({cnn.Zip(cnn.Skip(1),
    (p, n) => p * n).Sum()})");
    Console.WriteLine($"NN weights{" ",2}= {weights.Length}");
    Console.WriteLine($"SEED{" ",8}= {SEED:F0}");
    Console.WriteLine($"EPOCHS{" ",6}= {EPOCHS:F0}");
    Console.WriteLine($"BATCHSIZE{" ",3}= {BATCH:F0}");
    Console.WriteLine($"CNN LR{" ",6}= {LR:F3} | MLT = {FACTOR:F2}");
    Console.WriteLine($"NN LR{" ",7}= {LR:F3} | MLT = {FACTOR:F2}");
    Console.WriteLine($"Momentum{" ",4}= {MOMENTUM:F2}{" ",2}| MLT = {FACTOR:F2}");
    Console.WriteLine($"Dropout{" ",5}= {DROPOUT:F2}");
}

// next steps could be:
// 7.0 save cnn+nn
// 8.0 load cnn+nn

struct AutoData
{
    public byte[] labelsTraining, labelsTest;
    public float[] samplesTrainingF, samplesTestF;
    static float[] NormalizeData(byte[] samples) => samples.Select(s => s / 255f).ToArray();
    public AutoData(string path, Dataset datasetType)
    {
        byte[] test, training; // Define URLs and file paths based on dataset
        string trainDataUrl, trainLabelUrl, testDataUrl, testLabelUrl;
        string trainDataPath, trainLabelPath, testDataPath, testLabelPath;
        if (datasetType == Dataset.MNIST)
        {
            var baseMnistUrl = // Hardcoded URLs for MNIST data
            "https://github.com/grensen/gif_test/raw/master/MNIST_Data/";
            (trainDataUrl, trainLabelUrl, testDataUrl, testLabelUrl) = (
            $"{baseMnistUrl}train-images.idx3-ubyte",
            $"{baseMnistUrl}train-labels.idx1-ubyte",
            $"{baseMnistUrl}t10k-images.idx3-ubyte",
            $"{baseMnistUrl}t10k-labels.idx1-ubyte");
            (trainDataPath, trainLabelPath, testDataPath, testLabelPath) = (
            "trainData_MNIST", "trainLabel_MNIST",
            "testData_MNIST", "testLabel_MNIST");
        }
        else
        { // if (datasetType == DatasetType.FashionMNIST)
            var baseFashionUrl = // Hardcoded URLs for Fashion MNIST
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/";
            (trainDataUrl, trainLabelUrl, testDataUrl, testLabelUrl) = (
            $"{baseFashionUrl}train-images-idx3-ubyte.gz",
            $"{baseFashionUrl}train-labels-idx1-ubyte.gz",
            $"{baseFashionUrl}t10k-images-idx3-ubyte.gz",
            $"{baseFashionUrl}t10k-labels-idx1-ubyte.gz"); // Paths for Fashion MNIST data
            (trainDataPath, trainLabelPath, testDataPath, testLabelPath) = (
            "trainData_FashionMNIST", "trainLabel_FashionMNIST",
            "testData_FashionMNIST", "testLabel_FashionMNIST");
        }
        if (!File.Exists(Path.Combine(path, trainDataPath))
        || !File.Exists(Path.Combine(path, trainLabelPath))
        || !File.Exists(Path.Combine(path, testDataPath))
        || !File.Exists(Path.Combine(path, testLabelPath)))
        {
            Console.WriteLine($"Status: {datasetType} data not found");
            if (!Directory.Exists(path)) Directory.CreateDirectory(path);
            // padding bits: data = 16, labels = 8
            Console.WriteLine("Action: Downloading data from GitHub");
            training = DownloadAndExtract(trainDataUrl, datasetType).Skip(16).Take(60000 * 784).ToArray();
            labelsTraining = DownloadAndExtract(trainLabelUrl, datasetType).Skip(8).Take(60000).ToArray();
            test = DownloadAndExtract(testDataUrl, datasetType).Skip(16).Take(10000 * 784).ToArray();
            labelsTest = DownloadAndExtract(testLabelUrl, datasetType).Skip(8).Take(10000).ToArray();
            Console.WriteLine("Save path: " + path + "\n");
            File.WriteAllBytesAsync(Path.Combine(path, trainDataPath), training);
            File.WriteAllBytesAsync(Path.Combine(path, trainLabelPath), labelsTraining);
            File.WriteAllBytesAsync(Path.Combine(path, testDataPath), test);
            File.WriteAllBytesAsync(Path.Combine(path, testLabelPath), labelsTest);
        }
        else
        {
            Console.WriteLine($"Dataset: {datasetType} ({path})" + "\n");
            training = File.ReadAllBytes(Path.Combine(path, trainDataPath)).Take(60000 * 784).ToArray();
            labelsTraining = File.ReadAllBytes(Path.Combine(path, trainLabelPath)).Take(60000).ToArray();
            test = File.ReadAllBytes(Path.Combine(path, testDataPath)).Take(10000 * 784).ToArray();
            labelsTest = File.ReadAllBytes(Path.Combine(path, testLabelPath)).Take(10000).ToArray();
        }
        samplesTrainingF = NormalizeData(training); samplesTestF = NormalizeData(test);
    }
    static byte[] DownloadAndExtract(string url, Dataset datasetType)
    {
        using var client = new HttpClient();
        using var responseStream = client.GetStreamAsync(url).Result;
        using var ms = new MemoryStream();
        if (datasetType == Dataset.FashionMNIST) using (var gzipStream = new GZipStream(responseStream, CompressionMode.Decompress))
                gzipStream.CopyTo(ms);
        else responseStream.CopyTo(ms);
        return ms.ToArray();
    }
    public enum Dataset { MNIST, FashionMNIST }
}
