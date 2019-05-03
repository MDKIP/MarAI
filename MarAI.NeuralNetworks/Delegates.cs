using System;
using System.Collections.Generic;
using System.Text;

namespace MarAI.NeuralNetworks
{
    public delegate T FirstDimensionUnitBuilder<T>(long y);
    public delegate T SecondDimensionUnitBuilder<T>(long x, long y);
}
