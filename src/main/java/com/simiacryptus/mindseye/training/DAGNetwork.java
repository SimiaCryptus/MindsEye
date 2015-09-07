package com.simiacryptus.mindseye.training;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.NDArray;

import groovy.lang.Tuple2;

/***
 * Builds a network NNLayer components, assumed to form a directed acyclic graph with a single output.
 * Supplied builder methods designed to build linear sequence of units acting on the current output node.
 *
 * @author Andrew Charneski
 */
public class DAGNetwork extends NNLayer {
  private final class InputNode extends LazyResult {
    @Override
    protected NNResult[] eval(final EvaluationContext t) {
      return (NNResult[]) t.cache.get(inputHandle);
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.addProperty("target", inputHandle.toString());
      return json;
    }
  }

  private static final class UnaryNode extends LazyResult {
    private final NNLayer layer;
    private final LazyResult prevHead;

    private UnaryNode(NNLayer layer, LazyResult prevHead) {
      this.layer = layer;
      this.prevHead = prevHead;
    }

    @Override
    protected NNResult[] eval(final EvaluationContext ctx) {
      final NNResult[] input = prevHead.get(ctx);
      final NNResult output = layer.eval(ctx, input);
      return new NNResult[] { output };
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.add("layer", layer.getJson());
      json.add("prev", prevHead.toJson());
      return json;
    }
  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);

  private final java.util.LinkedHashMap<UUID, NNLayer> byId = new java.util.LinkedHashMap<>();
  private final java.util.HashMap<NNLayer, NNLayer> byPrev = new java.util.HashMap<>();
  private final java.util.HashMap<NNLayer, NNLayer> byNext = new java.util.HashMap<>();

  private LazyResult head = new InputNode();
  public final UUID inputHandle = UUID.randomUUID();

  public synchronized DAGNetwork add(final NNLayer layer) {
    NNLayer headLayer = getHeadLayer();
    this.byId.put(layer.getId(), layer);
    this.byPrev.put(headLayer, layer);
    this.byNext.put(layer, headLayer);
    this.setHead(new UnaryNode(layer, this.getHead()));
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... array) {
    evaluationContext.cache.put(this.inputHandle, array);
    return this.getHead().get(evaluationContext)[0];
  }

  public NNResult eval(final NDArray... array) {
    return eval(new EvaluationContext(), array);
  }

  public NNLayer get(final int i) {
    return this.byId.get(i);
  }

  @Override
  public List<NNLayer> getChildren() {
    return this.byId.values().stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId())).collect(Collectors.toList());
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("root", this.getHead().toJson());
    // for(NNLayer c : getChildren()){
    // json.add(c.getId(), c.getJson());
    // }
    return json;
  }

  public Tester trainer(final NDArray[][] samples) {
    return new Tester().setParams(this, samples);
  }


  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l->l.state().stream()).distinct().collect(Collectors.toList());
  }

  public void permute(UUID id, List<Tuple2<Integer, Integer>> permute) {
    NNLayer permutationLayer = byId.get(id);
    permutate_back(permutationLayer, permute);
    permutate_forward(permutationLayer, permute);
    
  }

  private void permutate_forward(NNLayer permutationLayer, List<Tuple2<Integer, Integer>> permute) {
    NNLayer next = this.byNext.get(permutationLayer);
    List<Tuple2<Integer, Integer>> passforward = next.permuteInput(permute);
    if(null != passforward) permutate_forward(next, passforward);
  }

  private void permutate_back(NNLayer permutationLayer, List<Tuple2<Integer, Integer>> permute) {
    NNLayer prev = this.byPrev.get(permutationLayer);
    List<Tuple2<Integer, Integer>> passback = prev.permuteOutput(permute);
    if(null != passback) permutate_back(prev, passback);
  }

  public LazyResult getHead() {
    return head;
  }

  public NNLayer getHeadLayer() {
    if(head instanceof UnaryNode) {
      return ((UnaryNode)head).layer;
    }
    return null;
  }

  public void setHead(LazyResult head) {
    this.head = head;
  }
}
