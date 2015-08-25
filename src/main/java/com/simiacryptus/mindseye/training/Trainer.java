package com.simiacryptus.mindseye.training;

import groovy.lang.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.math.NDArray;

public class Trainer {
  static final Logger log = LoggerFactory.getLogger(Trainer.class);

  private MutationTrainer inner = new MutationTrainer();

  public Trainer add(final PipelineNetwork pipelineNetwork, final NDArray[][] samples) {
    return add(new SupervisedTrainingParameters(pipelineNetwork, samples));
  }

  public Trainer add(final SupervisedTrainingParameters params) {
    this.getInner().getInner().getInner().getCurrent().add(params);
    return this;
  }

  public Tuple2<List<SupervisedTrainingParameters>, Double> getBest() {
    GradientDescentTrainer best = this.getInner().getBest();
    return new Tuple2<List<SupervisedTrainingParameters>, Double>(best.getCurrentNetworks(), best.error());
  }

  public Trainer setDynamicRate(final double d) {
    this.getInner().getInner().getInner().getCurrent().setRate(d);
    return this;
  }

  public Trainer setMaxDynamicRate(final double d) {
    this.getInner().getInner().setMaxRate(d);
    return this;
  }

  public Trainer setMinDynamicRate(final double d) {
    this.getInner().getInner().setMinRate(d);
    return this;
  }

  public Trainer setMutationAmount(final double d) {
    this.getInner().setMutationAmount(d);
    return this;
  }

  public Trainer setStaticRate(final double d) {
    this.getInner().getInner().setRate(d);
    return this;
  }

  public Trainer setVerbose(final boolean b) {
    this.getInner().setVerbose(b);
    return this;
  }

  public void train(final int i, final double d) {
    MutationTrainer inner = this.getInner();
    inner.setMaxIterations(i).setStopError(d);
    inner.train();
  }

  public long verifyConvergence(final int maxIter, final double convergence, final int reps) {
    return verifyConvergence(maxIter, convergence, reps, reps);
  }

  public long verifyConvergence(final int maxIter, final double convergence, final int reps, int minSuccess) {
    final long succeesses = IntStream.range(0, reps).parallel().filter(i -> testCopy(maxIter, convergence)).count();
    if (minSuccess > succeesses) throw new RuntimeException(String.format("%s out of %s converged", succeesses, reps));
    return succeesses;
  }

  public final List<Function<MutationTrainer, Void>> handler = new ArrayList<>();
  
  public boolean testCopy(final int maxIter, final double convergence) {
    boolean hasConverged = false;
    try {
      final MutationTrainer copy = Util.kryo().copy(this.getInner());
      final Double error = copy.setMaxIterations(maxIter).setStopError(convergence).train();
      handler.stream().forEach(h->h.apply(copy));
      hasConverged = error <= convergence;
      if (!hasConverged) {
        Trainer.log.debug(String.format("Not Converged: %s <= %s", error, convergence));
      }
    } catch (final Throwable e) {
      Trainer.log.debug("Not Converged", e);
    }
    return hasConverged;
  }

  public Trainer setMutationAmplitude(double d) {
    this.getInner().setMutationAmplitude(d);
    return this;
  }

  public MutationTrainer getInner() {
    return inner;
  }

  public void setInner(MutationTrainer inner) {
    this.inner = inner;
  }

}
