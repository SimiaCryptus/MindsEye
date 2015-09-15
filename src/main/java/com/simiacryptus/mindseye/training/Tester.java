package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.basic.RMSLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

/**
 * Encapsulates overall network architecture, training method and data.
 *
 * @author Andrew Charneski
 */
public class Tester {

  static final Logger log = LoggerFactory.getLogger(Tester.class);

  public final List<BiFunction<DAGNetwork, TrainingContext, Void>> handler = new ArrayList<>();

  private PopulationTrainer inner = new PopulationTrainer();

  private boolean parallel = true;

  TrainingContext trainingContext = new TrainingContext();

  public PopulationTrainer getInner() {
    return this.inner;
  }

  public boolean isParallel() {
    return this.parallel;
  }

  public Tester setDynamicRate(final double d) {
    getInner().getDynamicRateTrainer().getGradientDescentTrainer().setRate(d);
    return this;
  }

  public void setInner(final PopulationTrainer inner) {
    this.inner = inner;
  }

  public Tester setMaxDynamicRate(final double d) {
    getInner().getDynamicRateTrainer().setMaxRate(d);
    return this;
  }

  public Tester setMinDynamicRate(final double d) {
    getInner().getDynamicRateTrainer().setMinRate(d);
    return this;
  }

  public Tester setMutationAmplitude(final double d) {
    getInner().setAmplitude(d);
    return this;
  }

  public Tester setParallel(final boolean parallel) {
    this.parallel = parallel;
    return this;
  }

  public Tester setParams(final DAGNetwork pipelineNetwork, final NDArray[][] samples) {
    GradientDescentTrainer gradientDescentTrainer = getInner().getDynamicRateTrainer().getGradientDescentTrainer();
    DAGNetwork dagNetwork = new DAGNetwork();
    dagNetwork.add(pipelineNetwork);
    dagNetwork.add2(new RMSLayer());
    gradientDescentTrainer.setNet(dagNetwork);
    gradientDescentTrainer.setPredictionNode(dagNetwork.inputNode);
    gradientDescentTrainer.setMasterTrainingData(samples);
    return this;
  }

  public Tester setStaticRate(final double d) {
    getInner().getDynamicRateTrainer().getGradientDescentTrainer().setRate(d);
    return this;
  }

  public Tester setVerbose(final boolean b) {
    getInner().setVerbose(b);
    return this;
  }

  public void train(final double stopError, final TrainingContext trainingContext) throws TerminationCondition {
    getInner().getDynamicRateTrainer().setStopError(stopError);
    getInner().train(trainingContext);
  }

  private TrainingContext trainingContext() {
    return this.trainingContext;
  }

  public long verifyConvergence(final double convergence, final int reps) {
    return verifyConvergence(convergence, reps, reps);
  }

  public long verifyConvergence(final double convergence, final int reps, final int minSuccess) {
    IntStream range = IntStream.range(0, reps);
    if (isParallel()) {
      range = range.parallel();
    }
    final long succeesses = range.filter(i -> {
      final PopulationTrainer trainerCpy = Util.kryo().copy(getInner());
      final TrainingContext contextCpy = Util.kryo().copy(trainingContext());
      contextCpy.setTimeout(1, TimeUnit.MINUTES);
      return trainerCpy.test(convergence, contextCpy, this.handler);

    }).count();
    if (minSuccess > succeesses)
      throw new RuntimeException(String.format("%s out of %s converged", succeesses, reps));
    return succeesses;
  }

}
