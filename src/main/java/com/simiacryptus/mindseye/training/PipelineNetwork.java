package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Supplier;
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
  
  
  public static class LazyResult<T> {
    private transient volatile T value;
    private final Supplier<T> f;
    

    public LazyResult(Supplier<T> f) {
      super();
      this.f = f;
    }

    public T get() {
      if(null==this.value) {
       synchronized(this){
         if(null==this.value) {
          this.value = initialValue(); 
         }
       } 
      }
      return this.value;
    }

    protected final T initialValue() {
      return f.get();
    }
  }
  
  protected List<NNLayer> insertOrder = new ArrayList<NNLayer>();
  
  public PipelineNetwork add(final NNLayer layer) {
    this.insertOrder.add(layer);
    
    return this;
  }
  
  public NNResult eval(EvaluationContext evaluationContext, NNResult... array) {
    return eval(array);
  }

  public NNResult eval(final NDArray... inObj) {
    return eval(wrapInput(inObj));
  }

  public NNResult eval(final NNResult... inObj) {
    assert(1==inObj.length);
    NNResult r = inObj[0];
    for (final NNLayer l : this.insertOrder) {
      r = l.eval(new EvaluationContext(), r);
    }
    return r;
  }

  public NNLayer get(final int i) {
    return this.insertOrder.get(i);
  }
  
  @Override
  public String toString() {
    return "PipelineNetwork [" + this.insertOrder + "]";
  }
  
  public Trainer trainer(final NDArray[][] samples) {
    return new Trainer().set(this, samples);
  }
  
}