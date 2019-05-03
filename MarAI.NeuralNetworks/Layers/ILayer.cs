using System;
using System.Collections.Generic;
using System.Text;

namespace MarAI.NeuralNetworks
{
    public interface ILayer<U>
    {
        U[] Units { get; }
        long Size { get; }

        decimal[] CalculateOutputs(decimal[] inputs);
        void Adjust(decimal[] inputs, decimal[] errors);
    }
}
