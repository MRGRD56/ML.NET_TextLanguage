using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using ConsoleApp8.Models;
using Microsoft.ML;

namespace ConsoleApp8
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var dataView = mlContext.Data.LoadFromTextFile<Article>(@"Data\data.csv", ',', allowQuoting: true);
            
            var pipeline = 
                mlContext.Transforms.Conversion.MapValueToKey("Label", "Language")
                .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Text"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("Prediction", "PredictedLabel"));

            Console.WriteLine("Обучение модели...");
            var trainedModel = pipeline.Fit(dataView);
            Console.WriteLine("Обучение завершено.");

            var predictionEngine = mlContext.Model.CreatePredictionEngine<Article, ArticlePredication>(trainedModel);
            var prediction = predictionEngine.Predict(new Article
            {
                Text = "予測されたラベルを元に戻す方法の例を次に示します"
            });
            
            Console.WriteLine(prediction.Prediction);
        }

        private static string GetFileMD5(string fileName)
        {
            using var md5 = MD5.Create();
            using var stream = File.OpenRead(fileName);
            var hashBytes = md5.ComputeHash(stream);
            var result = BitConverter.ToString(hashBytes)
                .ToLower()
                .Replace("-", "");
            return result;
        }
    }
}