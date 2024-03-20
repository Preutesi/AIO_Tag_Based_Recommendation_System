using System.Collections.Generic;
using System;

namespace Word_2_Vec
{
    public class Word2Vec
    {
        public readonly Array<float> Vectors;
        public readonly string[] Text;
        public readonly Dictionary<string, int> Index;
        private Dictionary<string, int> _indexIgnoreCase;

        public Word2Vec(string[] words, Array<float> vectors)
        {
            int num = words.Length;
            Vectors = vectors;
            Text = words;
            Index = new Dictionary<string, int>(num);
            for (int i = 0; i < num; i++)
            {
                Index[words[i]] = i;
            }

            BuildIndexIgnoreCase();
        }

        private void BuildIndexIgnoreCase()
        {
            _indexIgnoreCase = new Dictionary<string, int>(StringComparer.InvariantCultureIgnoreCase);
            foreach (KeyValuePair<string, int> item in Index)
            {
                _indexIgnoreCase[item.Key] = item.Value;
            }
        }
    }
}
