package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.DeltaBuffer;
import com.simiacryptus.mindseye.learning.DeltaFlushBuffer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;

public class GradientDescentTrainer {

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);

  public static double geometricMean(final double[] error) {
    final double geometricMean = Math.exp(DoubleStream.of(error).filter(x -> 0 != x).map(Math::log).average().orElse(Double.POSITIVE_INFINITY));
    return geometricMean;
  }
  
  private SupervisedTrainingParameters currentNetwork = null;
  private double[] error;
  private double rate = 0.5;
  private boolean verbose = false;

  public GradientDescentTrainer() {
  }

  public GradientDescentTrainer set(final PipelineNetwork net, final NDArray[][] data) {
    return set(new SupervisedTrainingParameters(net, data));
  }

  public GradientDescentTrainer set(final SupervisedTrainingParameters params) {
    assert(null == currentNetwork);
    currentNetwork = params;
    return this;
  }

  protected double[] calcError(final List<NNResult> result) {
    List<Double> rms;
    {
      final SupervisedTrainingParameters currentNet = getCurrentNetwork();
      rms = IntStream.range(0, result.size()).parallel().mapToObj(sample -> {
        final NNResult eval = result.get(sample);
        final NDArray output = currentNet.getIdeal(eval, currentNet.getTrainingData()[sample][1]);
        final double err = eval.errRms(output);
        return err;
      }).collect(Collectors.toList());
    }
    return rms.stream().mapToDouble(x -> x).toArray();
  }

  public double error(TrainingContext trainingContext) {
    final double[] error = getError();
    if (null == error) {
      trainSet(trainingContext,null);
      return error(trainingContext);
    }
    final double returnValue = GradientDescentTrainer.geometricMean(error);
    assert Double.isFinite(returnValue);
    return returnValue;
  }

  protected List<NNResult> evalTrainingData(TrainingContext trainingContext) {
    SupervisedTrainingParameters params = getCurrentNetwork();
    return Stream.of(params.getTrainingData())
        .parallel()
        .map(sample -> {
          final NDArray input = sample[0];
          final NDArray output = sample[1];
          trainingContext.evaluations.increment();
          NDArray[] input1 = { input };
          final NNResult eval = params.getNet().eval(input1);
          assert eval.data.dim() == output.dim();
          return eval;
        }).collect(Collectors.toList());
  }

  public SupervisedTrainingParameters getCurrentNetwork() {
    return this.currentNetwork;
  }

  public synchronized double[] getError() {
    // if(null==this.error||0==this.error.length){
    // trainSet();
    // }
    return this.error;
  }

  public List<NNLayer> getLayers() {
    SupervisedTrainingParameters x = getCurrentNetwork();
    return x.getNet().insertOrder.stream().distinct().collect(Collectors.toList());
  }

  public PipelineNetwork getNetwork() {
    return this.currentNetwork.getNet();
  }

  public double getRate() {
    return this.rate;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  protected DeltaBuffer learn(final List<NNResult> results) {
    return learn(results, new DeltaBuffer());
  }

  protected DeltaBuffer learn(final List<NNResult> netresults, final DeltaBuffer buffer) {
    // Apply corrections
    {
      final SupervisedTrainingParameters currentNet = getCurrentNetwork();
      IntStream.range(0, netresults.size())
      // .parallel()
      .forEach(sample -> {
        final NNResult eval = netresults.get(sample);
        final NDArray output = currentNet.getIdeal(eval, currentNet.getTrainingData()[sample][1]);
        final NDArray delta2 = eval.delta(output);
        final LogNDArray log2 = delta2.log();
        LogNDArray delta = log2.scale(getRate());
        eval.feedback(delta, buffer);
      });
    };
    return buffer;
  }

  public GradientDescentTrainer setError(final double[] error) {
    this.error = error;
    return this;
  }

  public GradientDescentTrainer setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    return this;
  }

  public GradientDescentTrainer setVerbose(final boolean verbose) {
    if (verbose) {
      this.verbose = true;
    }
    this.verbose = verbose;
    return this;
  }

  public synchronized double[] trainSet(TrainingContext trainingContext, final double[] rates) {
    assert null != getCurrentNetwork();
    final List<NNResult> results = evalTrainingData(trainingContext);
    final double[] calcError = calcError(results);
    setError(calcError);
    final DeltaBuffer buffer = new DeltaBuffer();
    learn(results, buffer);
    final List<DeltaFlushBuffer> deltas = buffer.map.values().stream().collect(Collectors.toList());
    if (null != rates) {
      IntStream.range(0, buffer.map.size()).forEach(i -> {
        deltas.get(i).write(rates[i]);
      });
      final double[] validationError = calcError(evalTrainingData(trainingContext));
      if (GradientDescentTrainer.geometricMean(calcError) < GradientDescentTrainer.geometricMean(validationError)) {
        if (this.verbose) {
          GradientDescentTrainer.log.debug(String.format("Reverting: (%s)", Arrays.toString(calcError)));
        }
        IntStream.range(0, buffer.map.size()).forEach(i -> {
          deltas.get(i).write(-rates[i]);
        });
      } else {
        if (this.verbose) {
          GradientDescentTrainer.log.debug(String.format("Validating: (%s)", Arrays.toString(calcError)));
        }
      }
      setError(calcError);
    }
    return calcError;
  }

}
