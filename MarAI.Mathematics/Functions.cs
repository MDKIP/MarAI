using System;
using System.Collections.Generic;
using System.Text;

namespace MarAI.Mathematics
{
    static public class Functions
    {
        static public class Activation
        {
            static public decimal Identity(decimal x)
            {
                return x;
            }
            static public decimal BinaryStep(decimal x)
            {
                return x > 0 ? 1 : 0;
            }
        }
        static public class ActivationDerivatives
        {
            static public decimal Identity(decimal x)
            {
                return 1;
            }
        }
    }
}
