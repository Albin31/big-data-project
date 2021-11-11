import java.util.Arrays;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class RandomForest {

	static void init () {
      		MnistGenerateFile.loadFiles();
	}    
	
	static SparseVector getSVImage(int n) {
		int image[][] = MnistGenerateFile.getImage(n);	
		double vec[] = new double[MnistGenerateFile.rsize*MnistGenerateFile.csize];
		for (int r=0;r<MnistGenerateFile.rsize;r++)
				for (int c=0;c<MnistGenerateFile.csize;c++)
					vec[r*MnistGenerateFile.csize+c] = (double)image[r][c];
		DenseVector dv = new DenseVector(vec);
		SparseVector sv = dv.toSparse();
		return sv;
	}
	static void testPred(RandomForestClassificationModel model, int n) {
		int label = MnistGenerateFile.getLabel(n);
		SparseVector sv = getSVImage(n);
		double pred = model.predict(sv);
		System.out.format("Test a prediction (image %d): label is %d and prediction is %d\n", n, label,(int)pred);
	}
	
	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		
		SparkSession spark = SparkSession
				  .builder()
				  .appName("Data RandomForest")
				  .getOrCreate();
		
		// Load training data
		String path = "hdfs://master:54310/input/libsvm-mnist.txt";
		Dataset<Row> rawdata = spark.read().option("numFeatures", "784").format("libsvm").load(path);
		
		// Split the data into train and test
		Dataset<Row>[] splits = rawdata.randomSplit(new double[]{0.9, 0.1}, 1234L);
		Dataset<Row> train = splits[0];
		Dataset<Row> test = splits[1];
		
		// create the trainer and set its parameters
		RandomForestClassifier trainer = new RandomForestClassifier()
		  .setSeed(1234L);

	    	long t1 = System.currentTimeMillis(); 

		// train the model
		RandomForestClassificationModel model = trainer.fit(train);

		long t2 = System.currentTimeMillis(); 
	    
	    	System.out.println("======================"); 
	    	System.out.println("time in ms :"+(t2-t1)); 
	    	System.out.println("======================"); 
	    
		// compute accuracy on the test set
		Dataset<Row> result = model.transform(test);
		Dataset<Row> predictionAndLabels = result.select("prediction", "label");
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
		  .setMetricName("accuracy");

		System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
	
       		init();
		testPred(model, 9965);
		testPred(model, 9533);
		testPred(model, 9020);
		testPred(model, 9044);
		testPred(model, 9567);
		testPred(model, 9789);		
		
	}
}
