// https://github.com/grensen/cnn24
// .NET 8
// copy code
// then press to close functions: (ctrl + m + o)
// use Release mode, Debug is slow!

using System.IO.Compression;

// Glorot random weights init (template 0 and 1)
static float[] WeightsInitGlorot(int[] net, int seed)
{
    // get weights length (template 0)
    int len = 0;
    for (int i = 0; i < net.Length - 1; i++) // each layer
        for (int l = 0, left = net[i], right = net[i + 1]; l < left; l++) // input neurons
            for (int r = 0; r < right; r++) len++; // output neurons
                                                   // allocate memory
    float[] weights = new float[len];
    // set user seed
    Random rnd = new(seed);
    // Glorot random weights init (template 1)
    for (int i = 0, w = 0; i < net.Length - 1; i++) // each layer
    {
        int left = net[i], right = net[i + 1];
        float sd = MathF.Sqrt(6.0f / (net[i] + net[i + 1]));
        for (int l = 0; l < left; l++, w += right) // input neurons
        {
            for (int r = 0; r < right; r++) // output neurons
                weights[w + r] = rnd.NextSingle() * sd * 2 - sd;
        }
    }
    return weights;
}
// forward propagation (template 2)
static void FeedForward(float[] neurons, int[] net, float[] weights)
{
    for (int i = 0, j = 0, k = net[0], m = 0; i < net.Length - 1; i++) // each layer
    {
        int left = net[i], right = net[i + 1];
        for (int l = 0, w = m; l < left; l++, w += right) // input neurons
        {
            float n = neurons[j + l];
            if (n > 0) // ReLU pre-activation
                for (int r = 0; r < right; r++) // output neurons
                    neurons[k + r] += n * weights[w + r];
        }
        m += left * right; j += left; k += right;
    }
}
// backpropagation (template 3)
static void Backprop(float[] neurons, int[] net, float[] weights, float[] weightGradients)
{
    for (int i = net.Length - 2, j = neurons.Length - net[^1],
    k = neurons.Length, m = weights.Length; i >= 0; i--) // layers
    {
        int left = net[i], right = net[i + 1];
        m -= right * left; j -= left; k -= right;
        for (int l = 0, w = m; l < left; l++, w += right) // input neurons
        {
            float inputGradient = 0, n = neurons[j + l];
            if (n > 0) // ReLU derivative
                for (int r = 0; r < right; r++) // output neurons
                {
                    var gradient = neurons[k + r];
                    inputGradient += weights[w + r] * gradient;
                    weightGradients[w + r] += n * gradient;
                }
            neurons[j + l] = inputGradient;
        }
    }
}
// Stochastic Gradient Descent or Mini-batch-GD or Batch-GD (template 4)
static void Update(int[] net, float[] weights, float[] delta, float lr, float mom)
{
    for (int i = 0, w = 0; i < net.Length - 1; i++) // layers
    {
        int left = net[i], right = net[i + 1];
        for (int l = 0; l < left; l++, w += right) // input neurons
        {
            for (int r = 0; r < right; r++) // output neurons
            {
                weights[w + r] += delta[w + r] * lr;
                delta[w + r] *= mom;
            }
        }
    }
}
// helper functions
static float[] FeedSample(float[] samplesTrainingF, int x, int neuronLen)
{
    // neurons (input+hidden+output)
    float[] neurons = new float[neuronLen];
    // dataset id
    int sampleID = x * 784;
    // copy sample to input layer
    for (int i = 0; i < 784; i++)
        neurons[i] = samplesTrainingF[sampleID + i];
    return neurons;
}
// argmax and softmax
static int SoftArgMax(Span<float> neurons)
{
    // argmax prediction
    int id = 0;
    float max = neurons[0];
    for (int i = 1; i < neurons.Length; i++)
        if (neurons[i] > max)
        {
            max = neurons[i];
            id = i;
        }
    // softmax activation
    float scale = 0;
    for (int n = 0; n < neurons.Length; n++) // max trick
        scale += neurons[n] = MathF.Exp((neurons[n] - max));
    for (int n = 0; n < neurons.Length; n++)
        neurons[n] /= scale; // pseudo probabilities now
    return id; // return nn prediction
}
// target - output: distance between what we want minus what we get
static void ErrorGradient(Span<float> neurons, int target)
{
    for (int i = 0; i < neurons.Length; i++)
        neurons[i] = target == i ? 1 - neurons[i] : -neurons[i];
}
static void RunTraining(int[] net, float[] weights, float[] data, byte[] labels, int
BATCHSIZE, int EPOCHS, float LEARNINGRATE, float MOMENTUM)
{
    Console.WriteLine($"Training progress:");
    float[] weightGradients = new float[weights.Length];
    int neuronLen = 0;
    for (int i = 0; i < net.Length; i++) neuronLen += net[i];
    for (int epoch = 0, B = labels.Length / BATCHSIZE; epoch < EPOCHS; epoch++) // each epoch
    {
        int correct = 0;
        for (int b = 0; b < B; b++) // each batch
        {
            for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++) // each sample
            {
                float[] neurons = FeedSample(data, x, neuronLen);
                FeedForward(neurons, net, weights);
                var outs = neurons.AsSpan().Slice(neuronLen - net[^1], net[^1]);
                int prediction = SoftArgMax(outs); // reference output neurons
                int target = labels[x];
                correct += target == prediction ? 1 : 0;
                ErrorGradient(outs, target); // reference output neurons
                Backprop(neurons, net, weights, weightGradients);
            }
            Update(net, weights, weightGradients, LEARNINGRATE, MOMENTUM);
        }
        Console.WriteLine($"Epoch = {1 + epoch,2} | accuracy = {correct * 100.0 / (B * BATCHSIZE),5:F2}%");
    }
}
//
static void RunTesting(int[] net, float[] weights, float[] data, byte[] labels)
{
    int neuronLen = 0;
    for (int i = 0; i < net.Length; i++) neuronLen += net[i];
    int correct = 0;
    for (int x = 0; x < labels.Length; x++)
    { // each sample
        float[] neurons = FeedSample(data, x, neuronLen);
        FeedForward(neurons, net, weights);
        var outs = neurons.AsSpan().Slice(neuronLen - net[^1], net[^1]);
        int prediction = SoftArgMax(outs);
        int target = labels[x];
        correct += target == prediction ? 1 : 0;
    }
    Console.WriteLine($"\nTest accuracy = {(correct * 100.0 / labels.Length),6}%");
}
// run the code:
Console.WriteLine($"Begin basic neural network demo\n");
// get dataset
AutoData d = new(@"C:\basic_nn\", AutoData.Dataset.MNIST);
// define neural network
int[] net = { 784, 100, 100, 10 };
// get random weights
float[] weights = WeightsInitGlorot(net, 12345);
// nn training
RunTraining(net, weights, d.samplesTrainingF, d.labelsTraining, 100, 10, 0.001f, 0.5f);
// nn test
RunTesting(net, weights, d.samplesTestF, d.labelsTest);
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
