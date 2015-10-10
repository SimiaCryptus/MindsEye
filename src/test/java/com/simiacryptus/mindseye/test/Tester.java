package com.simiacryptus.mindseye.test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.training.DynamicRateTrainer;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;
import com.simiacryptus.mindseye.training.TrainingComponent;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

/**
 * Encapsulates overall network architecture, training method and data.
 *
 * @author Andrew Charneski
 */
public class Tester {

  static final Logger log = LoggerFactory.getLogger(Tester.class);

  public static DAGNetwork initPredictionNetwork(final NNLayer<?> predictor, final NNLayer<?> loss) {
    final DAGNetwork dagNetwork = new DAGNetwork();
    dagNetwork.add(predictor);
    dagNetwork.addLossComponent(loss);
    return dagNetwork;
  }

  protected DynamicRateTrainer dynamicTrainer;

  protected GradientDescentTrainer gradientTrainer;
  public final List<BiFunction<DAGNetwork, TrainingContext, Void>> handler = new ArrayList<>();
  private boolean parallel = true;
  private final TrainingContext trainingContext;

  public Tester() {
    super();
    initLayers();
    this.trainingContext = new TrainingContext().setTimeout(1, TimeUnit.MINUTES);
  }

  public DynamicRateTrainer getDynamicRateTrainer() {
    return this.dynamicTrainer;
  }

  public GradientDescentTrainer getGradientDescentTrainer() {
    return this.gradientTrainer;
  }

  public DAGNetwork getNet() {
    return this.dynamicTrainer.getNet();
  }

  public Tester init(final NDArray[][] samples, final NNLayer<DAGNetwork> pipelineNetwork, final NNLayer<?> lossLayer) {
    final DAGNetwork initPredictionNetwork = initPredictionNetwork(pipelineNetwork, lossLayer);
    this.gradientTrainer.setNet(initPredictionNetwork);
    this.gradientTrainer.setData(samples);
    return this;
  }

  public void initLayers() {
    this.gradientTrainer = new GradientDescentTrainer();
    this.dynamicTrainer = new DynamicRateTrainer(this.gradientTrainer);
  }

  public boolean isParallel() {
    return this.parallel;
  }

  public Tester setDynamicRate(final double d) {
    getGradientDescentTrainer().setRate(d);
    return this;
  }

  public Tester setMaxDynamicRate(final double d) {
    getDynamicRateTrainer().setMaxRate(d);
    return this;
  }

  public Tester setMinDynamicRate(final double d) {
    getDynamicRateTrainer().setMinRate(d);
    return this;
  }

  public Tester setParallel(final boolean parallel) {
    this.parallel = parallel;
    return this;
  }

  public Tester setStaticRate(final double d) {
    getGradientDescentTrainer().setRate(d);
    return this;
  }

  public Tester setVerbose(final boolean b) {
    getGradientDescentTrainer().setVerbose(b);
    getDynamicRateTrainer().setVerbose(b);
    return this;
  }

  public void train(final double stopError, final TrainingContext trainingContext) throws TerminationCondition {
    trainingContext.terminalErr = stopError;
    this.dynamicTrainer.step(trainingContext);
  }

  public TrainingContext trainingContext() {
    return this.trainingContext;
  }

  public long verifyConvergence(final double convergence, final int reps) {
    return verifyConvergence(convergence, reps, reps);
  }

  public long verifyConvergence(final double convergence, final int reps, final int minSuccess) {
    {
      final NDArray[][] trainingData = this.dynamicTrainer.getData();
      assert null != trainingData && 0 < trainingData.length;
    }
    IntStream range = IntStream.range(0, reps);
    if (isParallel()) {
      range = range.parallel();
    }
    final long succeesses = range.filter(i -> {
      final TrainingComponent trainerCpy;
      if (reps > 1) {
        synchronized (this.dynamicTrainer) {
          final NDArray[][] trainingData = this.dynamicTrainer.getData();
          assert null != trainingData && 0 < trainingData.length;
          this.dynamicTrainer.setData(null);
          trainerCpy = reps == 1 ? this.dynamicTrainer : Util.kryo().copy(this.dynamicTrainer);
          trainerCpy.setData(trainingData);
          this.dynamicTrainer.setData(trainingData);
        }
      } else {
        trainerCpy = this.dynamicTrainer;
        final NDArray[][] trainingData = this.dynamicTrainer.getData();
        assert null != trainingData && 0 < trainingData.length;
      }
      final TrainingContext trainingContext2 = trainingContext();
      final TrainingContext contextCpy = reps == 1 ? trainingContext2 : Util.kryo().copy(trainingContext2);
      // contextCpy.setTimeout(1, TimeUnit.MINUTES);
      boolean hasConverged = false;
      try {
        contextCpy.terminalErr = convergence;
        final double error = trainerCpy.step(contextCpy).finalError();
        final DAGNetwork net = trainerCpy.getNet();
        this.handler.stream().forEach(h -> h.apply(net, contextCpy));
        hasConverged = error <= convergence;
        if (!hasConverged) {
          Tester.log.debug(String.format("Not Converged: %s <= %s", error, convergence));
        }
      } catch (final Throwable e) {
        Tester.log.debug("Not Converged", e);
      }
      return hasConverged;
    }).count();
    if (minSuccess > succeesses)
      throw new RuntimeException(String.format("%s out of %s converged", succeesses, reps));
    return succeesses;
  }

}
