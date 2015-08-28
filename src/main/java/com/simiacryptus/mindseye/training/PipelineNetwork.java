package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.NDArray;

/***
 * Defines the fundamental structure of a network, currently only simple linear layout.
 * 
 * @author Andrew Charneski
 *
 */
public class PipelineNetwork extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(PipelineNetwork.class);
  
  protected List<NNLayer> layers = new ArrayList<NNLayer>();
  
  private double rate = 1.;
  
  public PipelineNetwork add(final NNLayer layer) {
    this.layers.add(layer);
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult array) {
    NNResult r = array;
    for (final NNLayer l : this.layers) {
      r = l.eval(r);
    }
    return r;
  }

  public NNLayer get(final int i) {
    return this.layers.get(i);
  }
  
  public double getRate() {
    return this.rate;
  }
  
  public PipelineNetwork setRate(final double rate) {
    this.rate = rate;
    return this;
  }
  
  @Override
  public String toString() {
    return "PipelineNetwork [" + this.layers + "]";
  }
  
  public Trainer trainer(final NDArray[][] samples) {
    return new Trainer().add(this, samples);
  }
  
}