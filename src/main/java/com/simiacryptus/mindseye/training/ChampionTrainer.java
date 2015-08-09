package com.simiacryptus.mindseye.training;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;

public class ChampionTrainer {

  private static final Logger log = LoggerFactory.getLogger(ChampionTrainer.class);
  
  public GradientDescentTrainer best = null;
  public GradientDescentTrainer current = null;
  private boolean verbose = false;

  public ChampionTrainer() {
    this(new GradientDescentTrainer());
  }

  public ChampionTrainer(GradientDescentTrainer current) {
    assert(null!=current);
    this.current = current;
  }
  
  public GradientDescentTrainer getBest() {
    return this.best;
  }

  public boolean isVerbose() {
    return this.verbose;
  }
  
  public void revert() {
    if(null != this.best)
    {
      if (isVerbose()) {
        ChampionTrainer.log.debug(String.format("Revert to best = %s", null==this.best?null:this.best.error()));
        // log.debug(String.format("Discarding %s", best.getFirst().get(0).getNet()));
      }
      this.current = Util.kryo().copy(this.best);
    }
  }

  public ChampionTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    this.current.setVerbose(verbose);
    return this;
  }

  public Double train() {
    final long startMs = System.currentTimeMillis();
    current.trainSet();
    updateBest();
    if (this.verbose)
    {
      ChampionTrainer.log.debug(String.format("Trained Error: %s (%s) with rate %s in %.03fs",
          current.error(), Arrays.toString(current.error), current.getRate(),  (System.currentTimeMillis() - startMs) / 1000.));
    }
    return this.current.error();
  }

  protected void updateBest() {
    if (Double.isFinite(current.error()) && (null == this.best || this.best.error() > current.error())) {
      if (isVerbose()) {
        ChampionTrainer.log.debug(String.format("New best Error %s > %s", current.error(), null == this.best ? "null" : this.best.error()));
        // log.debug(String.format("Best: %s", currentNetworks.get(0).getNet()));
      }
      this.best = Util.kryo().copy(this.current);
    }
  }

}
