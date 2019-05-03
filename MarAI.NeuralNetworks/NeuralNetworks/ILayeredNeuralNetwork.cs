using System;
using System.Collections.Generic;
using System.Text;

namespace MarAI.NeuralNetworks
{
    interface ILayeredNeuralNetwork : INeuralNetwork
    {
        object[] Layers { get; }
    }
}
