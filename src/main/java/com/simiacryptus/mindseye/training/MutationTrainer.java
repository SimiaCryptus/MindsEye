package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.Coordinate;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.util.Util;

public class MutationTrainer {

  private static final Logger log = LoggerFactory.getLogger(MutationTrainer.class);
  
  private int currentGeneration = 0;
  private final DynamicRateTrainer inner;
  private int maxIterations = 100;
  double mutationAmplitude = 5.;
  private double mutationFactor = .1;
  private double stopError = 0.1;
  private boolean verbose = false;

  public MutationTrainer() {
    this(new ChampionTrainer());
  }

  public MutationTrainer(final ChampionTrainer inner) {
    this.inner = new DynamicRateTrainer(inner);
  }

  public boolean continueTraining() {
    if (this.maxIterations < this.currentGeneration) {
      if (this.verbose) {
        MutationTrainer.log.debug("Reached max iterations: " + this.currentGeneration);
      }
      return false;
    }
    if (getInner().error() < this.stopError) {
      if (this.verbose) {
        MutationTrainer.log.debug("Reached convergence: " + getInner().error());
      }
      return false;
    }
    return true;
  }

  private double entropy(final BiasLayer l) {
    return 0;
  }

  private double entropy(final DenseSynapseLayer l, final Coordinate idx) {
    final NDArray weights = l.weights;
    final int[] dims = weights.getDims();
    final int columns = dims[0];
    final int rows = dims[1];
    final DoubleMatrix matrix = new DoubleMatrix(columns, rows, weights.getData()).transpose();
    // DoubleMatrix matrix = new DoubleMatrix(rows, columns, l.weights.getData());
    return IntStream.range(0, rows).filter(i -> i == idx.coords[1]).mapToDouble(i -> i).flatMap(i -> {
      return IntStream.range(0, rows).mapToDouble(j -> {
        final ArrayRealVector vi = new ArrayRealVector(matrix.getRow((int) i).toArray());
        if (vi.getNorm() <= 0.) return 0.;
        vi.unitize();
        final ArrayRealVector vj = new ArrayRealVector(matrix.getRow(j).toArray());
        if (vj.getNorm() <= 0.) return 0.;
        vj.unitize();
        return Math.acos(vi.cosine(vj));
      });
    }).average().getAsDouble();
  }

  public double error() {
    return getInner().getInner().getCurrent().error();
  }

  public GradientDescentTrainer getBest() {
    return getInner().getBest();
  }

  public int getGenerationsSinceImprovement() {
    return getInner().generationsSinceImprovement;
  }

  public DynamicRateTrainer getInner() {
    return this.inner;
  }

  private List<NNLayer> getLayers() {
    return getInner().getLayers();
  }
  
  public int getMaxIterations() {
    return this.maxIterations;
  }
  
  public double getMaxRate() {
    return getInner().getMaxRate();
  }
  
  public double getMinRate() {
    return getInner().minRate;
  }
  
  public double getMutationAmplitude() {
    return this.mutationAmplitude;
  }

  public double getMutationFactor() {
    return this.mutationFactor;
  }

  public List<PipelineNetwork> getNetwork() {
    return this.inner.getNetwork();
  }
  
  public double getRate() {
    return getInner().getRate();
  }
  
  public int getRecalibrationThreshold() {
    return getInner().recalibrationThreshold;
  }
  
  public double getStopError() {
    return this.stopError;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public int mutate(final BiasLayer l, final double amount) {
    final double[] a = l.bias;
    final Random random = Util.R.get();
    int sum = 0;
    for (int i = 0; i < a.length; i++)
    {
      if (random.nextDouble() < amount) {
        final double prev = a[i];
        final double prevEntropy = entropy(l);
        a[i] = randomWeight(l, random);
        final double nextEntropy = entropy(l);
        if (nextEntropy < prevEntropy) {
          a[i] = prev;
        } else {
          sum += 1;
        }
      }
    }
    return sum;
  }

  public int mutate(final DenseSynapseLayer l, final double amount) {
    final double[] a = l.weights.getData();
    final Random random = Util.R.get();
    return l.weights.coordStream().mapToInt(idx -> {
      final int i = idx.index;
      if (random.nextDouble() < amount) {
        final double prev = a[i];
        final double prevEntropy = entropy(l, idx);
        a[i] = randomWeight(l, random);
        final double nextEntropy = entropy(l, idx);
        if (nextEntropy < prevEntropy) {
          a[i] = prev;
          return 0;
        } else return 1;
      } else return 0;
    }).sum();
  }
  
  public int mutate(final double amount) {
    if (this.verbose) {
      MutationTrainer.log.debug(String.format("Mutating %s by %s", getInner(), amount));
    }
    final List<NNLayer> layers = getLayers();
    final int sum =
        layers.stream()
            .filter(l -> (l instanceof DenseSynapseLayer))
            .map(l -> (DenseSynapseLayer) l)
            .filter(l -> !l.isFrozen())
            .mapToInt(l -> mutate(l, amount))
            .sum() +
            layers.stream()
                .filter(l -> (l instanceof BiasLayer))
                .map(l -> (BiasLayer) l)
                .filter(l -> !l.isFrozen())
                .mapToInt(l -> mutate(l, amount))
                .sum();
    getInner().getInner().getCurrent().setError(null);
    return sum;
  }
  
  public void mutateBest() {
    getInner().generationsSinceImprovement = getInner().recalibrationThreshold - 1;
    getInner().getInner().revert();
    while (0 >= mutate(getMutationFactor())) {
    }
    getInner().lastCalibratedIteration = getInner().currentIteration;// - (this.recalibrationInterval + 2);
  }

  private double randomWeight(final BiasLayer l, final Random random) {
    return this.mutationAmplitude * random.nextGaussian() * 0.2;
  }
  
  public double randomWeight(final DenseSynapseLayer l, final Random random) {
    return this.mutationAmplitude * random.nextGaussian() / Math.sqrt(l.weights.getDims()[0]);
  }
  
  public MutationTrainer setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    getInner().generationsSinceImprovement = generationsSinceImprovement;
    return this;
  }
  
  public MutationTrainer setMaxIterations(final int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }
  
  public MutationTrainer setMaxRate(final double maxRate) {
    getInner().setMaxRate(maxRate);
    return this;
  }
  
  public MutationTrainer setMinRate(final double minRate) {
    getInner().minRate = minRate;
    return this;
  }
  
  public MutationTrainer setMutationAmount(final double mutationAmount) {
    getInner().setMutationFactor(mutationAmount);
    return this;
  }

  public MutationTrainer setMutationAmplitude(final double mutationAmplitude) {
    this.mutationAmplitude = mutationAmplitude;
    return this;
  }

  public void setMutationFactor(final double mutationRate) {
    this.mutationFactor = mutationRate;
  }

  public MutationTrainer setRate(final double rate) {
    getInner().setRate(rate);
    return this;
  }

  public MutationTrainer setRecalibrationThreshold(final int recalibrationThreshold) {
    getInner().recalibrationThreshold = recalibrationThreshold;
    return this;
  }
  
  public MutationTrainer setStopError(final double stopError) {
    getInner().setStopError(stopError);
    this.stopError = stopError;
    return this;
  }
  
  public MutationTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    getInner().setVerbose(verbose);
    return this;
  }
  
  public Double train() {
    final long startMs = System.currentTimeMillis();
    this.currentGeneration = 0;
    while (continueTraining())
    {
      if (0 == this.currentGeneration++) {
        mutate(1);
        mutate(1);
        mutate(1);
        mutate(1);
        mutate(1);
      } else {
        mutateBest();
      }
      getInner().trainToLocalOptimum();
      if (this.verbose)
      {
        MutationTrainer.log.debug(String.format("Trained Iteration %s Error: %s (%s) with rate %s",
            this.currentGeneration, getInner().error(), Arrays.toString(getInner().getInner().getCurrent().getError()), getInner().getInner().getCurrent()
                .getRate()));
      }
    }
    MutationTrainer.log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", getInner().error(),
        (System.currentTimeMillis() - startMs) / 1000.,
        this.currentGeneration));
    final GradientDescentTrainer best = getBest();
    return null == best ? Double.POSITIVE_INFINITY : best.error();
  }
}
