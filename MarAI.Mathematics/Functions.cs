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
            static public decimal Sigmoid(decimal x)
            {
                return (decimal)(1 / (1 + Math.Exp((double)-x)));
            }
            static public decimal Tanh(decimal x)
            {
                return (decimal)Math.Tanh((double)x);
            }
        }
        static public class ActivationDerivatives
        {
            static public decimal Identity(decimal x)
            {
                return 1;
            }
            static public decimal BinaryStep(decimal x)
            {
                return 1;
            }
            static public decimal Sigmoid(decimal x)
            {
                return x * (1 - x);
            }
            static public decimal Tanh(decimal x)
            {
                return 1 - (x*x);
            }
        }
    }
}
