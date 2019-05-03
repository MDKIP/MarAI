using MarAI.Mathematics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MarAI.NeuralNetworks
{
    public class SingleLayerNeuralNetwork : ILayeredNeuralNetwork
    {
        public SingleLayerNeuralNetwork(long inputsNumber, long outputsNumber, FirstDimensionUnitBuilder<IPerceptron> builder)
        {
            InputsNumber = inputsNumber;
            OutputsNumber = outputsNumber;

            Layer = new PerceptronLayer(OutputsNumber, builder);
            Layers = new object[] { Layer };
        }

        public PerceptronLayer Layer { get; private set; }
        public object[] Layers { get; private set; }
        public long InputsNumber { get; private set; }
        public long OutputsNumber { get; private set; }

        public void Backpropagation(decimal[] inputs, decimal[] answers)
        {
            Layer.Adjust(inputs, Layer.CalculateErrors(Layer.CalculateOutputs(inputs), answers));
        }
        public decimal[] Feedforward(decimal[] inputs)
        {
            return Layer.CalculateOutputs(inputs);
        }
        public void StartTraining(Dictionary<decimal[], decimal[]> dataset, long iterations)
        {
            int l = dataset.Count;
            for (long i = 0; i < iterations; i++)
            {
                var data = dataset.ElementAt(MathHelper.GetRandomInt(l));
                Backpropagation(data.Key, data.Value);
            }
        }
        public async Task StartTrainingAsync(Dictionary<decimal[], decimal[]> dataset, long iterations)
        {
            Task.Run(() => StartTraining(dataset, iterations));
        }
    }
}
