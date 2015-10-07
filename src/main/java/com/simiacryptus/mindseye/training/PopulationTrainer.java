package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dev.PermutationLayer;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

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
  private TrainingComponent inner;
  private int numberOfGenerations = 0;
  private int populationSize = 1;
  private boolean verbose = false;
  private NetInitializer netInitializer = new NetInitializer();

  protected PopulationTrainer() {
    super();
    inner = new DevelopmentTrainer();
  }

  public PopulationTrainer(TrainingComponent inner) {
    super();
    this.inner = inner;
  }

  private void align(final TrainingContext trainingContext, final List<TrainingComponent> population) {

    final List<List<List<double[]>>> signatures = population.stream().map(t -> {
      final DAGNetwork net = getNet();
      final List<PermutationLayer> pl = net.getChildren().stream() //
          .filter(x -> x instanceof PermutationLayer).map(x -> (PermutationLayer) x) //
          .collect(Collectors.toList());
      pl.stream().forEach(l -> l.record());
      final GradientDescentTrainer gd = new GradientDescentTrainer();
      gd.setData(getData());
      gd.setParallelTraining(false);
      gd.setNet(getNet());
      gd.setRate(0.);
      gd.step(trainingContext);
      //final double prevRate = gd.getRate();
      return pl.stream().map(l -> l.getRecord()).collect(Collectors.toList());
    }).collect(Collectors.toList());

    final int canonicalIndex = IntStream.range(0, population.size()) //
        .mapToObj(i -> new Tuple2<>(i, population.get(i))) //
        .sorted(Comparator.comparing(t -> t.getSecond().getError())) //
        .findFirst().get().getFirst();
    assert signatures.stream().allMatch(s -> s.size() == signatures.get(canonicalIndex).size());
    final List<PermutationLayer> syncLayers = population.get(canonicalIndex).getNet().getChildren().stream().filter(x -> x instanceof PermutationLayer)
        .map(x -> (PermutationLayer) x).collect(Collectors.toList());

    IntStream.range(0, signatures.size()).filter(i -> i != canonicalIndex).forEach(individual -> {
      IntStream.range(0, syncLayers.size()).forEach(layerIndex -> {
        final List<double[]> canonicalSignature = signatures.get(canonicalIndex).get(layerIndex);
        final List<double[]> individualSignature = signatures.get(individual).get(layerIndex);
        assert canonicalSignature.size() == individualSignature.size();
        if (0 < canonicalSignature.size()) {
          final List<Tuple2<Integer, Integer>> permute = findMapping(individualSignature, canonicalSignature);
          log.debug(String.format("Permutation in layer %s from %s to %s: %s", layerIndex, individual, canonicalIndex, permute));
          if (isAlignEnabled()) {
            final DAGNetwork net = population.get(individual).getNet();
            net.permute(syncLayers.get(layerIndex).getId(), permute);
          }
        }
      });
    });
  }

  @Override
  public NDArray[][] getData() {
    return inner.getData();
  }

  @Override
  public double getError() {
    return this.inner.getError();
  }

  @Override
  public DAGNetwork getNet() {
    return this.inner.getNet();
  }

  public int getNumberOfGenerations() {
    return this.numberOfGenerations;
  }

  public int getPopulationSize() {
    return this.populationSize;
  }

  public boolean isAlignEnabled() {
    return this.alignEnabled;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  private void measure(final TrainingContext trainingContext, final List<TrainingComponent> population) {
    if (1 >= population.size())
      return;
    population.stream().flatMapToDouble(a -> {
      final List<double[]> state1 = a.getNet().state();
      log.debug(String.format("Evaluating geometric alignment for %s (%s err)", a, a.getError()));
      return population.stream()
          // .filter(b->System.identityHashCode(b)<System.identityHashCode(a))
          .mapToDouble(b -> {
        final List<double[]> state2 = b.getNet().state();
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
      final List<double[]> state1 = a.getNet().state();
      log.debug(String.format("Evaluating arithmetic alignment for %s (%s err)", a, a.getError()));
      return population.stream()
          // .filter(b->System.identityHashCode(b)<System.identityHashCode(a))
          .mapToDouble(b -> {
        final List<double[]> state2 = b.getNet().state();
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

  private List<TrainingComponent> recombine(final List<TrainingComponent> progenators) {
    final List<TrainingComponent> nextGen = IntStream.range(0, getPopulationSize()).mapToObj(i -> {
      return Util.kryo().copy(this.inner);
    }).collect(Collectors.toList());
    final List<List<double[]>> progenators_state = progenators.stream().map(a -> a.getNet().state()).collect(Collectors.toList());
    nextGen.stream().forEach(a -> {
      final List<double[]> state = a.getNet().state();
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
    netInitializer.setAmplitude(amplitude);
    return this;
  }

  public TrainingComponent setNumberOfGenerations(final int numberOfGenerations) {
    this.numberOfGenerations = numberOfGenerations;
    return this;
  }

  public TrainingComponent setPopulationSize(final int populationSize) {
    this.populationSize = populationSize;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public TrainingStep step(final TrainingContext trainingContext) {
    return trainingContext.overallTimer.time(() -> {
      return train(trainingContext);
    });
  }

  private TrainingStep train(final TrainingContext trainingContext) {
    final long startMs = System.currentTimeMillis();
    double prevError = this.inner.getError();

    List<TrainingComponent> population = IntStream.range(0, getPopulationSize()).mapToObj(i -> {
      return Util.kryo().copy(this.inner);
    }).collect(Collectors.toList());

    population = population.stream().map(dynamicRateTrainer -> {
      final TrainingComponent dynamicRateTrainer1 = dynamicRateTrainer;
      netInitializer.initialize(dynamicRateTrainer1.getNet());
      dynamicRateTrainer1.reset();
      return dynamicRateTrainer;
    }).collect(Collectors.toList());

    for (int generation = 0; generation <= getNumberOfGenerations(); generation++) {
      population = population.stream().parallel().map(dynamicRateTrainer -> {
        trainIndividual(trainingContext, dynamicRateTrainer);
        if (this.verbose) {
          PopulationTrainer.log.info(String.format("Completed training to %.5f in %.03fs - %s", dynamicRateTrainer.getError(),
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
        final List<TrainingComponent> progenators = population.stream().sorted(Comparator.comparing(x -> x.getError())).limit(3)
            .collect(Collectors.toList());
        population = recombine(progenators);
      }
    }

    this.inner = population.stream().sorted(Comparator.comparing(x -> x.getError())).findFirst().get();
    double nextError = this.inner.getError();
    return new TrainingStep(prevError, nextError, true);
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
  public void reset() {
    inner.reset();
  }
}
