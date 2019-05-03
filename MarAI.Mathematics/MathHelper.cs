using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace MarAI.Mathematics
{
    static public class MathHelper
    {
        static MathHelper()
        {
            _rng = RandomNumberGenerator.Create();
        }

        static private RandomNumberGenerator _rng;
        static private Random _rand;

        static public int GetRandomInt()
        {
            byte[] b = new byte[3];
            _rng.GetBytes(b);

            _rand = new Random(b[0] * b[1] * b[2]);
            return _rand.Next();
        }
        static public int GetRandomInt(int max)
        {
            byte[] b = new byte[3];
            _rng.GetBytes(b);

            _rand = new Random(b[0] * b[1] * b[2]);
            return _rand.Next(max);
        }
        static public int GetRandomInt(int min, int max)
        {
            byte[] b = new byte[3];
            _rng.GetBytes(b);

            _rand = new Random(b[0] * b[1] * b[2]);
            return _rand.Next(min, max);
        }
        static public decimal GetRandomDecimal()
        {
            byte[] b = new byte[3];
            _rng.GetBytes(b);

            _rand = new Random(b[0] * b[1] * b[2]);
            return (decimal)_rand.NextDouble();
        }
        public static decimal Map(this decimal value, decimal fromSource, decimal toSource, decimal fromTarget, decimal toTarget)
        {
            return (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget;
        }
    }
}
