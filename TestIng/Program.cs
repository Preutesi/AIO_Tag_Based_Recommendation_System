using Neural_Network;
using Word_2_Vec;
using Recommendation;

var matrix = NeuralNetwork.Normal(0, 1, 1, 100);
char[] chars = "$%#@!*abcdefghijklmnopqrstuvwxyz1234567890?;:ABCDEFGHIJKLMNOPQRSTUVWXYZ^&".ToCharArray();

List<string[]> tags = [];
Random r = new();
Random r2 = new();

for (int k = 0; k < 20; k++)
{
    int reps = r.Next(10, 500);
    string[] tmpV = new string[reps];
    for (int i = 0; i < reps; i++)
    {
        string blt = "";
        for (int j = 0; j < 20; j++)
            blt += chars[r2.Next(chars.Length)];
        tmpV[i] = blt;
    }
    tags.Add(tmpV);
}

foreach (string[] tag in tags)
{
    Word2Vec _w2c1 = new([string.Join(" ", tag)], matrix);
}

Suggestion s = new([tags[0], tags[1]],tags);
var x = s.GetSuggestions();
Console.Read();