package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.layers.NNLayer;

public class ChampionTrainer {

  private static final Logger log = LoggerFactory.getLogger(ChampionTrainer.class);
  
  private GradientDescentTrainer best = null;
  private GradientDescentTrainer current = null;
  private boolean verbose = false;

  public ChampionTrainer() {
    this(new GradientDescentTrainer());
  }

  public ChampionTrainer(final GradientDescentTrainer current) {
    assert null != current;
    this.setCurrent(current);
  }
  
  public GradientDescentTrainer getBest() {
    return this.best;
  }

  public boolean isVerbose() {
    return this.verbose;
  }
  
  public void revert() {
    GradientDescentTrainer best = this.getBest();
    if (null != best)
    {
      if (isVerbose()) {
        ChampionTrainer.log.debug(String.format("Revert to best = %s", null == best ? null : best.error()));
        // log.debug(String.format("Discarding %s", best.getFirst().get(0).getNet()));
      }
      this.setCurrent(Util.kryo().copy(best));
    }
  }

  public ChampionTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    this.getCurrent().setVerbose(verbose);
    return this;
  }

  public Double step() {
    final long startMs = System.currentTimeMillis();
    this.getCurrent().trainSet();
    updateBest();
    if (this.verbose)
    {
      ChampionTrainer.log.debug(String.format("Trained Error: %s (%s) with rate %s*%s in %.03fs",
          this.getCurrent().error(), Arrays.toString(this.getCurrent().getError()), this.getCurrent().getRate(), Arrays.toString(this.getCurrent().getRates()),
          (System.currentTimeMillis() - startMs) / 1000.));
    }
    return this.getCurrent().error();
  }

  protected void updateBest() {
    GradientDescentTrainer best = this.getBest();
    if (Double.isFinite(this.getCurrent().error()) && (null == best || best.error() > this.getCurrent().error())) {
      if (isVerbose()) {
        ChampionTrainer.log.debug(String.format("New best Error %s > %s", this.getCurrent().error(), null == best ? "null" : best.error()));
        // log.debug(String.format("Best: %s", currentNetworks.get(0).getNet()));
      }
      this.setBest(Util.kryo().copy(this.getCurrent()));
    }
  }

  public GradientDescentTrainer getCurrent() {
    return current;
  }

  public ChampionTrainer setCurrent(GradientDescentTrainer current) {
    this.current = current;
    return this;
  }

  public ChampionTrainer setBest(GradientDescentTrainer best) {
    this.best = best;
    return this;
  }

  public List<NNLayer> getLayers() {
    return getCurrent().getLayers();
  }

}
