package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext.LazyResult;

/***
 * Builds a linear pipeline of NNLayer components, applied in sequence
 * 
 * @author Andrew Charneski
 */
public class PipelineNetwork extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(PipelineNetwork.class);
  
  private final List<NNLayer> children = new ArrayList<NNLayer>();
  
  public final UUID inputHandle = UUID.randomUUID();
  public LazyResult<NNResult[]> head = new LazyResult<NNResult[]>() {
    @Override
    protected NNResult[] initialValue(EvaluationContext t) {
      return (NNResult[]) t.cache.get(inputHandle);
    }
  };
  
  public synchronized PipelineNetwork add(final NNLayer layer) {
    children.add(layer);
    LazyResult<NNResult[]> prevHead = head;
    head = new LazyResult<NNResult[]>() {
      @Override
      protected NNResult[] initialValue(EvaluationContext ctx) {
        NNResult[] input = prevHead.get(ctx);
        NNResult output = layer.eval(ctx, input);
        return new NNResult[] { output };
      }
    };
    return this;
  }
  
  public NNResult eval(EvaluationContext evaluationContext, NNResult... array) {
    evaluationContext.cache.put(inputHandle, array);
    return head.get(evaluationContext)[0];
  }
  
  public NNLayer get(final int i) {
    return this.getChildren().get(i);
  }
  
  @Override
  public String toString() {
    return "PipelineNetwork [" + this.getChildren() + "]";
  }
  
  public Tester trainer(final NDArray[][] samples) {
    return new Tester().set(this, samples);
  }
  
  public NNResult eval(NDArray... array) {
    return eval(new EvaluationContext(), array);
  }
  
  public List<NNLayer> getChildren() {
    return children.stream()
        .flatMap(l->l.getChildren().stream())
        .sorted(Comparator.comparing(l->l.getId()))
        .collect(Collectors.toList());
  }
  
}