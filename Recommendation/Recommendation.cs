using System;
using System.Collections.Generic;
using System.Linq;
using Word_2_Vec;
using Neural_Network;

namespace Recommendation
{
    public class Suggestion
    {
        private static List<double[]> _w2cLiked = new List<double[]>();
        private static List<double[]> _w2cContainer = new List<double[]>();

        public Suggestion(List<string[]> likedTags, List<string[]> allTags)
        {
            NeuralNetwork.SetSeed(12);
            (_w2cLiked, _w2cContainer) = GetValues(likedTags, allTags);
        }

        public Dictionary<int, double> GetSuggestions(double aboveOf = 0.00)
        {
            Dictionary<int, double> d = new Dictionary<int, double>();
            int counter = 0;
            foreach (double[] tv in _w2cContainer)
            {
                var distances = CalculateDistances(_w2cLiked, tv);
                double[] normalizedDistances = NormalizeDistances(distances);
                d.Add(counter, 1 - normalizedDistances.Average());
                counter++;
            }
            var sortedEntries = from entry in d orderby entry.Value descending select entry;
            Dictionary<int, double> result = new Dictionary<int, double>();
            foreach (var entry in sortedEntries)
            {
                result.Add(entry.Key, entry.Value);
            }
            return result;
        }

        private static List<double> CalculateDistances(List<double[]> doubleList, double[] target)
        {
            return doubleList.Select(arr => Distance(arr, target)).ToList();
        }

        private static double Distance(double[] arr1, double[] arr2)
        {
            if (arr1 == null || arr2 == null || arr1.Length != arr2.Length)
            {
                throw new ArgumentException("Both input arrays should be non-null and have equal length.");
            }

            double sumOfSquares = 0;
            int len = arr1.Length;
            for (int i = 0; i < len; i++)
            {
                double diff = arr1[i] - arr2[i];
                sumOfSquares += diff * diff;
            }
            return Math.Sqrt(sumOfSquares);
        }

        private static double[] NormalizeDistances(IEnumerable<double> distances)
        {
            double maxDistance = distances.Max();
            return distances.Select(d => 1 - d / maxDistance).ToArray();
        }

        static (List<double[]>, List<double[]>) GetValues(List<string[]> likedTags, List<string[]> allTags)
        {
            List<double[]> _w2cLiked = GetArrays(likedTags);
            List<double[]> _w2cContainer = GetArrays(allTags);

            return (_w2cLiked, _w2cContainer);
        }

        static List<double[]> GetArrays(List<string[]> tags)
        {
            List<double[]> rValues = new List<double[]>();
            foreach (string[] tag in tags)
            {
                var matrix = NeuralNetwork.Normal(0, 1, 1, 100);
                string[] tmpJoin = new string[1];
                tmpJoin[0] = string.Join(" ", tag);
                Word2Vec _w2c1 = new Word2Vec(tmpJoin, matrix);

                List<double> temp = new List<double>();
                for (int j = 0; j < 100; j++)
                {
                    temp.Add((double)_w2c1.Vectors[j]);
                }
                rValues.Add(temp.ToArray());
            }
            return rValues;
        }
    }
}
