using Microsoft.ML.Data;

namespace ConsoleApp8.Models
{
    public class Article
    {
        [LoadColumn(0)]
        public string Text { get; set; }
        
        [LoadColumn(1)]
        public string Language { get; set; }
    }
}