using System;
using System.Collections.Generic;
using System.Text;

namespace MarAI.NeuralNetworks
{
    public class PerceptronLayer : ILayer<IPerceptron>
    {
        public PerceptronLayer(long size, FirstDimensionUnitBuilder<IPerceptron> builder)
        {
            Size = size;
            Units = new IPerceptron[Size];
            for (long y = 0; y < Size; y++)
            {
                Units[y] = builder.Invoke(y);
            }
        }

        public IPerceptron[] Units { get; private set; }
        public long Size { get; private set; }

        public void Adjust(decimal[] inputs, decimal[] errors)
        {
            for (long y = 0; y < Size; y++)
            {
                Units[y].AdjustParametersByError(inputs, errors[y]);
            }
        }
        public decimal[] CalculateOutputs(decimal[] inputs)
        {
            decimal[] outputs = new decimal[Size];
            for (long y = 0; y < Size; y++)
            {
                outputs[y] = Units[y].CalculateOutput(inputs);
            }
            return outputs;
        }
        public decimal[] CalculateDerivatives(decimal[] outputs)
        {
            decimal[] derivatives = new decimal[Size];
            for (long i = 0; i < Size; i++)
            {
                derivatives[i] = Units[i].ActivationDerivative.Invoke(outputs[i]);
            }
            return derivatives;
        }
        public decimal[] CalculateErrors(decimal[] outputs, decimal[] answers)
        {
            decimal[] errors = new decimal[Size];
            for (long i = 0; i < Size; i++)
            {
                errors[i] = answers[i] - outputs[i];
            }
            return errors;
        }
        public decimal[] CalculateGradients(decimal[] currentErrors, decimal[] currentDerivatives)
        {
            decimal[] gradients = new decimal[Size];
            for (long i = 0; i < Size; i++)
            {
                gradients[i] = currentErrors[i] * currentDerivatives[i];
            }
            return gradients;
        }
        public decimal[] CalculateGradients(PerceptronLayer nextLayer, decimal[] nextLayerGradients, decimal[] currentDerivatives)
        {
            decimal[] gradients = new decimal[Size];
            for (long c = 0; c < Size; c++)
            {
                decimal gradient = 0;
                for (long n = 0; n < nextLayer.Size; n++)
                {
                    gradient += nextLayerGradients[n] * nextLayer.Units[n].Weights[c];
                }
                gradient *= currentDerivatives[c];
                gradients[c] = gradient;
            }
            return gradients;
        }
    }
}
