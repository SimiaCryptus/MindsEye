package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Coordinate;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;

public class MutationTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(MutationTrainer.class);

  private int currentGeneration = 0;
  private final DynamicRateTrainer inner;
  private int maxIterations = 1000;
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
    if (this.getInner().error() < this.stopError) {
      if (this.verbose) {
        MutationTrainer.log.debug("Reached convergence: " + this.getInner().error());
      }
      return false;
    }
    return true;
  }
  
  public double error() {
    return this.getInner().getInner().getCurrent().error();
  }
  
  public GradientDescentTrainer getBest() {
    return this.getInner().getBest();
  }
  
  public int getGenerationsSinceImprovement() {
    return this.getInner().generationsSinceImprovement;
  }
  
  public DynamicRateTrainer getInner() {
    return this.inner;
  }
  
  public int getMaxIterations() {
    return this.maxIterations;
  }
  
  public double getMaxRate() {
    return this.getInner().maxRate;
  }
  
  public double getMinRate() {
    return this.getInner().minRate;
  }
  
  public double getMutationFactor() {
    return this.mutationFactor;
  }

  public double getRate() {
    return this.getInner().getRate();
  }

  public int getRecalibrationThreshold() {
    return this.getInner().recalibrationThreshold;
  }

  public double getStopError() {
    return this.stopError;
  }

  public boolean isVerbose() {
    return this.verbose;
  }
  
  public int mutate(final BiasLayer l, final double amount) {
    final double[] a = l.bias;
    Random random = Util.R.get();
    int sum = 0;
    for (int i = 0; i < a.length; i++)
    {
      if (random.nextDouble() < amount) {
        double prev = a[i];
        double prevEntropy = entropy(l);
        a[i] = randomWeight(l, random);
        double nextEntropy = entropy(l);
        if(nextEntropy < prevEntropy) {
          a[i] = prev;
        } else {
          sum += 1;
        }
      }
    }
    return sum;
  }
  
  private double randomWeight(BiasLayer l, Random random) {
    return mutationAmplitude * random.nextGaussian() * 0.0;
  }

  private double entropy(BiasLayer l) {
    return 0;
  }

  public int mutate(final DenseSynapseLayer l, final double amount) {
    final double[] a = l.weights.getData();
    Random random = Util.R.get();
    return l.weights.coordStream().mapToInt(idx->{
      int i = idx.index;
      if (random.nextDouble() < amount) {
        double prev = a[i];
        double prevEntropy = entropy(l, idx);
        a[i] = randomWeight(l, random);
        double nextEntropy = entropy(l, idx);
        if(nextEntropy < prevEntropy) {
          a[i] = prev;
          return 0;
        } else {
          return 1;
        }
      } else {
        return 0;
      }
    }).sum();
  }

  public double randomWeight(final DenseSynapseLayer l, Random random) {
    return mutationAmplitude * random.nextGaussian() / Math.sqrt(l.weights.getDims()[0]);
  }
  
  private double entropy(DenseSynapseLayer l, Coordinate idx) {
    NDArray weights = l.weights;
    int[] dims = weights.getDims();
    int columns = dims[0];
    int rows = dims[1];
    DoubleMatrix matrix = new DoubleMatrix(columns, rows, weights.getData()).transpose();
    //DoubleMatrix matrix = new DoubleMatrix(rows, columns, l.weights.getData());
    return IntStream.range(0, rows).filter(i->i==(int)idx.coords[1]).mapToDouble(i->i).flatMap(i->{
      return IntStream.range(0, rows).mapToDouble(j->{
        ArrayRealVector vi = new ArrayRealVector(matrix.getRow((int) i).toArray());
        if(vi.getNorm() <= 0.) return 0.;
        vi.unitize();
        ArrayRealVector vj = new ArrayRealVector(matrix.getRow(j).toArray());
        if(vj.getNorm() <= 0.) return 0.;
        vj.unitize();
        return Math.acos(vi.cosine(vj));
      });
    }).average().getAsDouble();
  }
  
  public int mutate(final double amount) {  
    if (this.verbose) {
      MutationTrainer.log.debug(String.format("Mutating %s by %s", this.getInner(), amount));
    }
    List<NNLayer> layers = this.getLayers();
    int sum = 
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
    this.getInner().getInner().getCurrent().setError(null);
    return sum;
  }
  
  private List<NNLayer> getLayers() {
    return getInner().getLayers();
  }

  double mutationAmplitude = 2.;

  public void mutateBest() {
    this.getInner().generationsSinceImprovement = this.getInner().recalibrationThreshold - 1;
    this.getInner().getInner().revert();
    while(0 >= mutate(getMutationFactor())) {}
    this.getInner().lastCalibratedIteration = this.getInner().currentIteration;// - (this.recalibrationInterval + 2);
  }
  
  public MutationTrainer setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    this.getInner().generationsSinceImprovement = generationsSinceImprovement;
    return this;
  }

  public MutationTrainer setMaxIterations(final int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }

  public MutationTrainer setMaxRate(final double maxRate) {
    this.getInner().maxRate = maxRate;
    return this;
  }

  public MutationTrainer setMinRate(final double minRate) {
    this.getInner().minRate = minRate;
    return this;
  }

  public MutationTrainer setMutationAmount(final double mutationAmount) {
    this.getInner().setMutationFactor(mutationAmount);
    return this;
  }

  public void setMutationFactor(final double mutationRate) {
    this.mutationFactor = mutationRate;
  }

  public MutationTrainer setRate(final double rate) {
    this.getInner().setRate(rate);
    return this;
  }
  
  public MutationTrainer setRecalibrationThreshold(final int recalibrationThreshold) {
    this.getInner().recalibrationThreshold = recalibrationThreshold;
    return this;
  }
  
  public MutationTrainer setStopError(final double stopError) {
    getInner().setStopError(stopError);
    this.stopError = stopError;
    return this;
  }
  
  public MutationTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    this.getInner().setVerbose(verbose);
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
      this.getInner().trainToLocalOptimum();
      if (this.verbose)
      {
        MutationTrainer.log.debug(String.format("Trained Iteration %s Error: %s (%s) with rate %s",
            this.currentGeneration, this.getInner().error(), Arrays.toString(this.getInner().getInner().getCurrent().getError()), this.getInner().getInner().getCurrent().getRate()));
      }
    }
    MutationTrainer.log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", this.getInner().error(),
        (System.currentTimeMillis() - startMs) / 1000.,
        this.currentGeneration));
    final GradientDescentTrainer best = this.getBest();
    return null == best ? Double.POSITIVE_INFINITY : best.error();
  }

  public double getMutationAmplitude() {
    return this.mutationAmplitude;
  }

  public MutationTrainer setMutationAmplitude(final double mutationAmplitude) {
    this.mutationAmplitude = mutationAmplitude;
    return this;
  }

  public List<PipelineNetwork> getNetwork() {
    return inner.getNetwork();
  }
}
