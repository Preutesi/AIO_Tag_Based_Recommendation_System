using System;
using System.Runtime.CompilerServices;
using System.Security.Policy;

namespace Neural_Network
{
    public class NeuralNetwork
    {
        public static Random Random = new Random(1236);

        public static Array<T> Fill<T>(Func<T> f, Array<T> result)
        {
            return Array_.ElementwiseOp(result, delegate (int n, T[] r, int offR, int strideR)
            {
                for (int i = 0; i < n; i++)
                {
                    r[offR] = f();
                    offR += strideR;
                }
            });
        }

        public static Array<T> Fill<T>(Func<T> f, params int[] shape)
        {
            return Fill(f, new Array<T>(shape));
        }

        public static Array<float> Normal(float mean, float std, params int[] shape)
        {
            return NeuralNetwork.Fill(() => NextNormalF(mean, std), shape);
        }

        public static Random SetSeed(int seed)
        {
            Random = new Random(seed);
            return Random;
        }
        private static double NextNormal(double mean, double std)
        {
            double d = Random.NextDouble();
            double num = Random.NextDouble();
            double num2 = Math.Sqrt(-2.0 * Math.Log(d)) * Math.Sin(Math.PI * 2.0 * num);
            return mean + std * num2;
        }

        private static float NextNormalF(float mean, float std)
        {
            return (float)NextNormal(mean, std);
        }
    }
}

[Flags]
public enum Flags
{
    Transposed = 1,
    NotContiguous = 2
}

public static class Array_
{
    public static Array<T1> ElementwiseOp<T1>(Array<T1> a, Action<int, T1[], int, int> op)
    {
        ElementwiseOp(0, a, 0, op);
        return a;
    }

    public static void ElementwiseOp<T1>(int axis, Array<T1> a, int offseta, Action<int, T1[], int, int> op)
    {
        int num = a.Shape.Length - 1;
        if (num == -1)
        {
            op(1, a.Values, offseta + a.Offset, 1);
            return;
        }

        while (num != 0 && a.Shape[num] == 1)
        {
            num--;
        }

        if (axis == num)
        {
            op(a.Shape[axis], a.Values, offseta + a.Offset, a.Stride[axis]);
            return;
        }

        for (int i = 0; i < a.Shape[axis]; i++)
        {
            ElementwiseOp(axis + 1, a, offseta, op);
            offseta += a.Stride[axis];
        }
    }
}

public static class StridedExtension
{
    public static int ComputeSize(int[] shape)
    {
        int num = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            num *= shape[i];
        }

        return num;
    }

    public static int[] ComputeStride(int[] shape, int[] result = null)
    {
        result = result ?? new int[shape.Length];
        int num = result.Length - 1;
        if (num == -1)
        {
            return result;
        }

        result[num] = 1;
        for (int num2 = num - 1; num2 >= 0; num2--)
        {
            result[num2] = result[num2 + 1] * shape[num2 + 1];
        }

        return result;
    }
}
public class Array<Type> : Strided<Type>
{
    public Type[] Values;

    public Array(int[] shape, Type[] values, int offset, int[] stride)
    {
        Shape = shape;
        Values = values;
        Offset = offset;
        Stride = stride;
        Flags = (CheckTransposed() ? Flags.Transposed : ((Flags)0)) | ((!CheckContiguous()) ? Flags.NotContiguous : ((Flags)0));
    }

    public Array(int[] shape, Type[] values)
        : this(shape, values, 0, StridedExtension.ComputeStride(shape))
    {
    }

    public Array(params int[] shape)
        : this(shape, new Type[StridedExtension.ComputeSize(shape)])
    {
    }

    private bool CheckContiguous()
    {
        bool result = true;
        for (int i = 0; i < base.NDim - 1; i++)
        {
            if (Stride[i] != Shape[i + 1] * Stride[i + 1])
            {
                result = false;
            }
        }

        return result;
    }

    private bool CheckTransposed()
    {
        bool result = false;
        for (int i = 0; i < base.NDim - 1; i++)
        {
            if (Stride[i] < Stride[i + 1])
            {
                result = true;
            }
        }

        return result;
    }

    public Type this[int i0]
    {
        get
        {
            return Values[i0];
        }
        set
        {
            Values[i0] = value;
        }
    }
}

public class Strided<Type>
{
    public Flags Flags;
    public int Offset;
    public int[] Shape;
    public int[] Stride;
    public int NDim
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return Shape.Length;
        }
    }

    public int Size => StridedExtension.ComputeSize(Shape);
}
