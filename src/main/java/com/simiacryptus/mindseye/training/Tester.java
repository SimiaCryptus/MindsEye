package com.simiacryptus.mindseye.training;

import groovy.lang.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

/**
 * Encapsulates overall network architecture, training method and data.
 * 
 * @author Andrew Charneski
 *
 */
public class Tester {
  
  static final Logger log = LoggerFactory.getLogger(Tester.class);
  
  public final List<BiFunction<MutationTrainer, TrainingContext, Void>> handler = new ArrayList<>();
  
  private MutationTrainer inner = new MutationTrainer();
  
  public Tester set(final PipelineNetwork pipelineNetwork, final NDArray[][] samples) {
    return set(new SupervisedTrainingParameters(pipelineNetwork, samples));
  }
  
  public Tester set(final SupervisedTrainingParameters params) {
    getInner().getInner().getInner().getCurrent().set(params);
    return this;
  }
  
  public Tuple2<SupervisedTrainingParameters, Double> getBest(TrainingContext trainingContext) {
    final GradientDescentTrainer best = getInner().getBest();
    return new Tuple2<SupervisedTrainingParameters, Double>(null == best ? null : best.getCurrentNetwork(), null == best ? null : best.error(trainingContext));
  }
  
  public MutationTrainer getInner() {
    return this.inner;
  }
  
  public Tester setDynamicRate(final double d) {
    getInner().getInner().getInner().getCurrent().setRate(d);
    return this;
  }
  
  public void setInner(final MutationTrainer inner) {
    this.inner = inner;
  }
  
  public Tester setMaxDynamicRate(final double d) {
    getInner().getInner().setMaxRate(d);
    return this;
  }
  
  public Tester setMinDynamicRate(final double d) {
    getInner().getInner().setMinRate(d);
    return this;
  }
  
  public Tester setMutationAmount(final double d) {
    getInner().setMutationAmount(d);
    return this;
  }
  
  public Tester setMutationAmplitude(final double d) {
    getInner().setMutationAmplitude(d);
    return this;
  }
  
  public Tester setStaticRate(final double d) {
    getInner().getInner().setRate(d);
    return this;
  }
  
  public Tester setVerbose(final boolean b) {
    getInner().setVerbose(b);
    return this;
  }

  public boolean testCopy(final int maxIter, final double convergence) {
    boolean hasConverged = false;
    try {
      final MutationTrainer copy = Util.kryo().copy(getInner());
      TrainingContext trainingContext = trainingContext();
      final Double error = trainingContext.overallTimer.time(()->{
        return copy.setMaxIterations(maxIter).setStopError(convergence).train(trainingContext);
      });
      this.handler.stream().forEach(h -> h.apply(copy,trainingContext));
      hasConverged = error <= convergence;
      if (!hasConverged) {
        Tester.log.debug(String.format("Not Converged: %s <= %s", error, convergence));
      }
    } catch (final Throwable e) {
      Tester.log.debug("Not Converged", e);
    }
    return hasConverged;
  }

  public void train(final int i, final double d, TrainingContext trainingContext) throws TerminationCondition {
    final MutationTrainer inner = getInner();
    inner.setMaxIterations(i).setStopError(d);
    inner.train(trainingContext);
  }
  
  private TrainingContext trainingContext() {
    return new TrainingContext();
  }

  public long verifyConvergence(final int maxIter, final double convergence, final int reps) {
    return verifyConvergence(maxIter, convergence, reps, reps);
  }
  
  public long verifyConvergence(final int maxIter, final double convergence, final int reps, final int minSuccess) {
    final long succeesses = IntStream.range(0, reps).parallel().filter(i -> testCopy(maxIter, convergence)).count();
    if (minSuccess > succeesses) throw new RuntimeException(String.format("%s out of %s converged", succeesses, reps));
    return succeesses;
  }
  
}
