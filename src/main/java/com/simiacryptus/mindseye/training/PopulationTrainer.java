package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.Coordinate;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dev.PermutationLayer;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class PopulationTrainer implements TrainingComponent {

  private static final Logger log = LoggerFactory.getLogger(PopulationTrainer.class);

  private static List<Tuple2<Integer, Integer>> findMapping(final List<double[]> from, final List<double[]> to) {
    final int dim = from.get(0).length;
    assert from.stream().allMatch(x -> dim == x.length);
    assert to.stream().allMatch(x -> dim == x.length);

    final double[][] vectorsFrom = IntStream.range(0, dim).mapToObj(i -> from.stream().mapToDouble(x -> x[i]).toArray()).toArray(i -> new double[i][]);
    final double[][] vectorsTo = IntStream.range(0, dim).mapToObj(i -> to.stream().mapToDouble(x -> x[i]).toArray()).toArray(i -> new double[i][]);

    final NDArray covariance = new NDArray(dim, dim);
    IntStream.range(0, dim).mapToObj(fromIndex -> {
      return IntStream.range(0, dim).mapToDouble(toIndex -> {
        double dot = 0;
        double magFrom = 0;
        double magTo = 0;
        final double[] vfrom = vectorsFrom[fromIndex];
        final double[] vto = vectorsTo[toIndex];
        for (int k = 0; k < vfrom.length; k++) {
          final double vkfrom = vfrom[k];
          final double vkto = vto[k];
          dot += vkfrom * vkto;
          magFrom += vkfrom * vkfrom;
          magTo += vkto * vkto;
        }
        dot /= Math.sqrt(magFrom);
        dot /= Math.sqrt(magTo);
        covariance.set(new int[] { fromIndex, toIndex }, dot);
        return dot;
      }).toArray();
    }).toArray(i -> new double[i][]);
    // log.debug(String.format("Covariance: %s", covariance));

    final ArrayList<Tuple2<Integer, Integer>> list = new java.util.ArrayList<>();
    final java.util.Set<Integer> toIndexes = new java.util.HashSet<>(IntStream.range(0, dim).mapToObj(x -> x).collect(Collectors.toList()));
    for (int fromIndex = 0; fromIndex < dim; fromIndex++) {
      final int _fromIndex = fromIndex;
      final int toIndex = toIndexes.stream().filter(x -> toIndexes.contains(x)) //
          .sorted(java.util.Comparator.comparing(i -> -covariance.get(_fromIndex, i))) //
          .findFirst().get();
      toIndexes.remove(toIndex);
      list.add(new Tuple2<Integer, Integer>(fromIndex, toIndex));
    }
    assert dim == list.size();
    return list;
  }

  private boolean alignEnabled = true;
  private double initAmplitude = 5.;
  private DynamicRateTrainer inner = new DynamicRateTrainer();
  private int numberOfGenerations = 0;
  private int populationSize = 1;
  private boolean verbose = false;

  private void align(final TrainingContext trainingContext, final List<DynamicRateTrainer> population) {
    final List<List<List<double[]>>> signatures = population.stream().map(t -> {
      GradientDescentTrainer gd = t.getGradientDescentTrainer();
      
      final DAGNetwork net = gd.getNet();
      final List<PermutationLayer> pl = net.getChildren().stream() //
          .filter(x -> x instanceof PermutationLayer).map(x -> (PermutationLayer) x) //
          .collect(Collectors.toList());
      pl.stream().forEach(l -> l.record());
      
      double prevRate = gd.getRate();
      gd.setRate(0);
      gd.setParallelTraining(false);
      gd.step(trainingContext);
      gd.setRate(prevRate);
      gd.setParallelTraining(true);
      
      return pl.stream().map(l -> l.getRecord()).collect(Collectors.toList());
    }).collect(Collectors.toList());

    final int canonicalIndex = IntStream.range(0, population.size()) //
        .mapToObj(i -> new Tuple2<>(i, population.get(i))) //
        .sorted(Comparator.comparing(t -> t.getSecond().getGradientDescentTrainer().getError())) //
        .findFirst().get().getFirst();
    assert signatures.stream().allMatch(s -> s.size() == signatures.get(canonicalIndex).size());
    final List<PermutationLayer> syncLayers = population.get(canonicalIndex).getGradientDescentTrainer().getNet().getChildren()
        .stream().filter(x -> x instanceof PermutationLayer).map(x -> (PermutationLayer) x).collect(Collectors.toList());

    IntStream.range(0, signatures.size()).filter(i -> i != canonicalIndex).forEach(individual -> {
      IntStream.range(0, syncLayers.size()).forEach(layerIndex -> {
        final List<double[]> canonicalSignature = signatures.get(canonicalIndex).get(layerIndex);
        final List<double[]> individualSignature = signatures.get(individual).get(layerIndex);
        assert canonicalSignature.size() == individualSignature.size();
        if (0 < canonicalSignature.size()) {
          final List<Tuple2<Integer, Integer>> permute = findMapping(individualSignature, canonicalSignature);
          log.debug(String.format("Permutation in layer %s from %s to %s: %s", layerIndex, individual, canonicalIndex, permute));
          if (isAlignEnabled()) {
            final DAGNetwork net = population.get(individual).getGradientDescentTrainer().getNet();
            net.permute(syncLayers.get(layerIndex).getId(), permute);
          }
        }
      });
    });
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
    // DoubleMatrix matrix = new DoubleMatrix(rows, columns,
    // l.weights.getData());
    return IntStream.range(0, rows).filter(i -> i == idx.coords[1]).mapToDouble(i -> i).flatMap(i -> {
      return IntStream.range(0, rows).mapToDouble(j -> {
        final ArrayRealVector vi = new ArrayRealVector(matrix.getRow((int) i).toArray());
        if (vi.getNorm() <= 0.)
          return 0.;
        vi.unitize();
        final ArrayRealVector vj = new ArrayRealVector(matrix.getRow(j).toArray());
        if (vj.getNorm() <= 0.)
          return 0.;
        vj.unitize();
        return Math.acos(vi.cosine(vj));
      });
    }).average().getAsDouble();
  }

  public DynamicRateTrainer getDynamicRateTrainer() {
    return this.inner;
  }

  public DAGNetwork getNet() {
    return getDynamicRateTrainer().getGradientDescentTrainer().getNet();
  }

  public int getNumberOfGenerations() {
    return this.numberOfGenerations;
  }

  public int getPopulationSize() {
    return this.populationSize;
  }

  private int initialize(final BiasLayer l) {
    final double[] a = l.bias;
    final Random random = Util.R.get();
    int sum = 0;
    for (int i = 0; i < a.length; i++) {
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
    return sum;
  }

  private int initialize(final DenseSynapseLayer l) {
    final double[] a = l.weights.getData();
    final Random random = Util.R.get();
    return l.weights.coordStream().mapToInt(idx -> {
      final int i = idx.index;
      final double prev = a[i];
      final double prevEntropy = entropy(l, idx);
      a[i] = randomWeight(l, random);
      final double nextEntropy = entropy(l, idx);
      if (nextEntropy < prevEntropy) {
        a[i] = prev;
        return 0;
      } else
        return 1;
    }).sum();
  }

  private void initialize(final DynamicRateTrainer dynamicRateTrainer) {
    if (this.verbose) {
      PopulationTrainer.log.debug(String.format("Initialize %s", dynamicRateTrainer));
    }
    final List<NNLayer<?>> layers = dynamicRateTrainer.getGradientDescentTrainer().getNet().getChildren();
    for (int i = 0; i < 5; i++) {
      layers.stream().filter(l -> (l instanceof DenseSynapseLayer)).map(l -> (DenseSynapseLayer) l).filter(l -> !l.isFrozen()).forEach(this::initialize);
      layers.stream().filter(l -> (l instanceof BiasLayer)).map(l -> (BiasLayer) l).filter(l -> !l.isFrozen()).forEach(this::initialize);
    }
    dynamicRateTrainer.getGradientDescentTrainer().setError(Double.NaN);
  }

  public boolean isAlignEnabled() {
    return this.alignEnabled;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  private void measure(final TrainingContext trainingContext, final List<DynamicRateTrainer> population) {
    if (1 >= population.size())
      return;
    population.stream().flatMapToDouble(a -> {
      final List<double[]> state1 = a.getGradientDescentTrainer().getNet().state();
      log.debug(String.format("Evaluating geometric alignment for %s (%s err)", a, a.getGradientDescentTrainer().getError()));
      return population.stream()
          // .filter(b->System.identityHashCode(b)<System.identityHashCode(a))
          .mapToDouble(b -> {
        final List<double[]> state2 = b.getGradientDescentTrainer().getNet().state();
        assert state1.size() == state2.size();
        double sum = 0.;
        int cnt = 0;
        for (int i = 0; i < state1.size(); i++) {
          final double[] a1 = state1.get(i);
          final double[] a2 = state2.get(i);
          assert a1.length == a2.length;
          for (int j = 0; j < a1.length; j++) {
            final double abs = Math.abs(Math.log(Math.abs(Math.abs(a1[j] - a2[j]) / a2[j])));
            if (Double.isFinite(abs)) {
              sum += abs;
              cnt++;
            }
          }
        }
        final double avg = Math.exp(sum / cnt);
        log.debug(String.format("Geometric Alignment between %s and %s: %s", a, b, avg));
        return avg;
      });
    }).average().getAsDouble();

    population.stream().flatMapToDouble(a -> {
      final List<double[]> state1 = a.getGradientDescentTrainer().getNet().state();
      log.debug(String.format("Evaluating arithmetic alignment for %s (%s err)", a, a.getGradientDescentTrainer().getError()));
      return population.stream()
          // .filter(b->System.identityHashCode(b)<System.identityHashCode(a))
          .mapToDouble(b -> {
        final List<double[]> state2 = b.getGradientDescentTrainer().getNet().state();
        assert state1.size() == state2.size();
        double sum = 0.;
        int cnt = 0;
        for (int i = 0; i < state1.size(); i++) {
          final double[] a1 = state1.get(i);
          final double[] a2 = state2.get(i);
          assert a1.length == a2.length;
          for (int j = 0; j < a1.length; j++) {
            final double abs = Math.abs(a1[j] - a2[j]);
            if (Double.isFinite(abs)) {
              sum += abs;
              cnt++;
            }
          }
        }
        final double avg = sum / cnt;
        log.debug(String.format("Arithmetic alignment between %s and %s: %s", a, b, avg));
        return avg;
      });
    }).average().getAsDouble();
  }

  protected double randomWeight(final BiasLayer l, final Random random) {
    return this.initAmplitude * random.nextGaussian() * 0.2;
  }

  protected double randomWeight(final DenseSynapseLayer l, final Random random) {
    return this.initAmplitude * random.nextGaussian() / Math.sqrt(l.weights.getDims()[0]);
  }

  private List<DynamicRateTrainer> recombine(final List<DynamicRateTrainer> progenators) {
    final List<DynamicRateTrainer> nextGen = IntStream.range(0, getPopulationSize()).mapToObj(i -> {
      return Util.kryo().copy(getDynamicRateTrainer());
    }).collect(Collectors.toList());
    final List<List<double[]>> progenators_state = progenators.stream().map(a -> a.getGradientDescentTrainer().getNet().state()).collect(Collectors.toList());
    nextGen.stream().forEach(a -> {
      final List<double[]> state = a.getGradientDescentTrainer().getNet().state();
      for (int i = 0; i < state.size(); i++) {
        final double[] a1 = state.get(i);
        for (int j = 0; j < a1.length; j++) {
          a1[j] = progenators_state.get(Util.R.get().nextInt(progenators_state.size())).get(i)[j];
        }
      }
    });
    return nextGen;
  }

  public TrainingComponent setAlignEnabled(final boolean align) {
    this.alignEnabled = align;
    return this;
  }

  public TrainingComponent setAmplitude(final double amplitude) {
    this.initAmplitude = amplitude;
    return this;
  }

  public TrainingComponent setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    getDynamicRateTrainer().generationsSinceImprovement = generationsSinceImprovement;
    return this;
  }

  public TrainingComponent setNumberOfGenerations(final int numberOfGenerations) {
    this.numberOfGenerations = numberOfGenerations;
    return this;
  }

  public PopulationTrainer setPopulationSize(final int populationSize) {
    this.populationSize = populationSize;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    this.verbose = verbose;
    getDynamicRateTrainer().setVerbose(verbose);
    return this;
  }

  public double step(final TrainingContext trainingContext) {
    final Double error = trainingContext.overallTimer.time(() -> {
      return train(trainingContext);
    });
    return error;
  }

  private Double train(final TrainingContext trainingContext) {
    final long startMs = System.currentTimeMillis();

    List<DynamicRateTrainer> population = IntStream.range(0, getPopulationSize()).mapToObj(i -> {
      return Util.kryo().copy(getDynamicRateTrainer());
    }).collect(Collectors.toList());

    population = population.stream().map(dynamicRateTrainer -> {
      initialize(dynamicRateTrainer);
      return dynamicRateTrainer;
    }).collect(Collectors.toList());

    for (int generation = 0; generation <= getNumberOfGenerations(); generation++) {
      population = population.stream().parallel().map(dynamicRateTrainer -> {
        trainIndividual(trainingContext, dynamicRateTrainer);
        if (this.verbose) {
          PopulationTrainer.log.info(String.format("Completed training to %.5f in %.03fs - %s", dynamicRateTrainer.getGradientDescentTrainer().getError(),
              (System.currentTimeMillis() - startMs) / 1000., trainingContext));
        }
        return dynamicRateTrainer;
      }).collect(Collectors.toList());

      measure(trainingContext, population);
      if (generation < getNumberOfGenerations()) {
        align(trainingContext, population);
        if (isAlignEnabled()) {
          log.debug("Re-alignment:");
          align(trainingContext, population);
          log.debug("Re-alignment:");
          align(trainingContext, population);
          measure(trainingContext, population);
        }
        final List<DynamicRateTrainer> progenators = population.stream().sorted(Comparator.comparing(x -> x.getGradientDescentTrainer().getError())).limit(3)
            .collect(Collectors.toList());
        population = recombine(progenators);
      }
    }

    this.inner = population.stream().sorted(Comparator.comparing(x -> x.getGradientDescentTrainer().getError())).findFirst().get();
    return getDynamicRateTrainer().getGradientDescentTrainer().getError();
  }

  private void trainIndividual(final TrainingContext trainingContext, final TrainingComponent dynamicRateTrainer) {
    try {
      trainingContext.mutations.increment();
      dynamicRateTrainer.step(trainingContext);
    } catch (final TerminationCondition e) {
      PopulationTrainer.log.debug("Terminated training", e);
    }
  }

  @Override
  public double getError() {
    return getDynamicRateTrainer().getError();
  }
}
