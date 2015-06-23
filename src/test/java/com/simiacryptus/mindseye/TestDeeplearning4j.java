package com.simiacryptus.mindseye;

import org.junit.Test;


import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

public class TestDeeplearning4j {
  private static final Logger log = LoggerFactory.getLogger(TestDeeplearning4j.class);

  @Test
  public void test() throws IOException{

    final int numRows = 28;
    final int numColumns = 28;
    int outputNum = 10;
    int numSamples = 100;
    int batchSize = 10;
    int iterations = 5;
    int splitTrainNum = (int) (batchSize*.8);
    //int seed = 123;
    int listenerFreq = iterations/5;
    DataSet mnist;
    SplitTestAndTrain trainTest;
    DataSet trainInput;
    List<INDArray> testInput = new ArrayList<>();
    List<INDArray> testLabels = new ArrayList<>();

    log.info("Load data....");
    DataSetIterator mnistIter = new MnistDataSetIterator(batchSize,numSamples);

    log.info("Build model....");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .nIn(numRows * numColumns)
            .nOut(outputNum)
            //.seed(seed)
            .batchSize(batchSize)
            .iterations(iterations)
            .weightInit(WeightInit.UNIFORM)
            .activationFunction("sigmoid")
            .filterSize(8, 1, numRows, numColumns)
            .optimizationAlgo(OptimizationAlgorithm.LBFGS)
            .constrainGradientToUnitNorm(true)
            .list(3)
            .hiddenLayerSizes(50)
            .inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns))
            .preProcessor(1, new ConvolutionPostProcessor())
            .override(0, new ConfOverride() {
                public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                    builder.layer(new ConvolutionLayer());
                    builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                    builder.featureMapSize(9, 9);
                }
            }).override(1, new ConfOverride() {
                public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                    builder.layer(new SubsamplingLayer());
                }
            }).override(2, new ClassifierOverride())
            .build();
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    log.info("Train model....");
    model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
//    model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));
    while(mnistIter.hasNext()) {
        mnist = mnistIter.next();
        mnist.normalizeZeroMeanZeroUnitVariance();
        trainTest = mnist.splitTestAndTrain(splitTrainNum); // train set that is the result
        trainInput = trainTest.getTrain(); // get feature matrix and labels for training
        testInput.add(trainTest.getTest().getFeatureMatrix());
        testLabels.add(trainTest.getTest().getLabels());
        model.fit(trainInput);
    }

    log.info("Evaluate weights....");
    for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
        INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        log.info("Weights: " + w);
    }

    log.info("Evaluate model....");
    Evaluation eval = new Evaluation();
    for(int i = 0; i < testInput.size(); i++) {
        INDArray output = model.output(testInput.get(i));
        eval.eval(testLabels.get(i), output);
    }

    log.info(eval.stats());
    log.info("****************Example finished********************");

  }
}
