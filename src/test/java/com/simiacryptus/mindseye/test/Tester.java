package com.simiacryptus.mindseye.test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.SupervisedNetwork;
import com.simiacryptus.util.lang.KryoUtil;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.training.DynamicRateTrainer;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;
import com.simiacryptus.mindseye.training.TrainingComponent;

/**
 * Encapsulates common training and testing setup
 *
 * @author Andrew Charneski
 */
public class Tester {

	static final Logger log = LoggerFactory.getLogger(Tester.class);

  public DynamicRateTrainer dynamicTrainer;
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

  public Tester init(final Tensor[][] samples, final NNLayer<DAGNetwork> pipelineNetwork, final NNLayer<?> lossLayer) {
    this.gradientTrainer.setNet(new SupervisedNetwork(pipelineNetwork, lossLayer));
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
    IntStream range = IntStream.range(0, reps);
    if (isParallel()) {
      range = range.parallel();
    }
    final long succeesses = range.filter(i -> {
      if(reps==1){
        return trainTo(convergence);
      } else {
        final TrainingComponent trainer = copy(this.dynamicTrainer);
        final TrainingContext trainingContext = KryoUtil.kryo().copy(trainingContext());
        return trainTo(trainer, trainingContext, convergence);
      }
    }).count();
    if (minSuccess > succeesses) {
      throw new RuntimeException(String.format("%s out of %s converged", succeesses, reps));
    }
    return succeesses;
  }

  public boolean trainTo(final double convergence) {
    return trainTo(this.dynamicTrainer, trainingContext(), convergence);
  }

  protected boolean trainTo(final TrainingComponent trainer, final TrainingContext trainingContext, final double convergence) {
    boolean hasConverged = false;
    try {
      trainingContext.terminalErr = convergence;
      final double error = trainer.step(trainingContext).finalError();
      final DAGNetwork net = trainer.getNet();
      this.handler.stream().forEach(h -> h.apply(net, trainingContext));
      hasConverged = error <= convergence;
      if (!hasConverged) {
        Tester.log.debug(String.format("Not Converged: %s <= %s", error, convergence));
      }
    } catch (final Throwable e) {
      Tester.log.debug("Not Converged", e);
    }
    return hasConverged;
  }

  public static TrainingComponent copy(DynamicRateTrainer trainer) {
    synchronized (trainer) {
      final Tensor[][] trainingData = trainer.getTrainingData();
      assert null != trainingData && 0 < trainingData.length;
      trainer.setData(null);
      DynamicRateTrainer copy = KryoUtil.kryo().copy(trainer);
      copy.setData(trainingData);
      trainer.setData(trainingData);
      return copy;
    }
  }

}
