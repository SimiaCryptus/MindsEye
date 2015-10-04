package com.simiacryptus.mindseye.net.dag;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;
import groovy.lang.Tuple2;

/***
 * Builds a network NNLayer components, assumed to form a directed acyclic graph
 * with a single output. Supplied builder methods designed to build linear
 * sequence of units acting on the current output node.
 *
 * @author Andrew Charneski
 */
public class DAGNetwork extends NNLayer<DAGNetwork> {
  public final class BinaryNode extends LazyResult<NNResult> {
    private final UUID layer;
    private final LazyResult<NNResult> left;
    private final LazyResult<NNResult> right;

    private BinaryNode(final NNLayer<?> layer, final LazyResult<NNResult> left, final LazyResult<NNResult> right) {
      this.layer = layer.getId();
      this.left = left;
      this.right = right;
    }

    @Override
    protected NNResult eval(final EvaluationContext ctx) {
      final NNResult inputL = (NNResult) this.left.get(ctx);
      final NNResult inputR = (NNResult) this.right.get(ctx);
      final NNResult output = DAGNetwork.this.byId.get(this.layer).eval(ctx, inputL, inputR);
      return output;
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.add("layer", DAGNetwork.this.byId.get(this.layer).getJson());
      json.add("left", this.left.toJson());
      json.add("right", this.right.toJson());
      return json;
    }
  }

  private final class InputNode extends LazyResult<NNResult[]> {
    @Override
    protected NNResult[] eval(final EvaluationContext t) {
      return (NNResult[]) t.cache.get(DAGNetwork.this.inputHandle);
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.addProperty("target", DAGNetwork.this.inputHandle.toString());
      return json;
    }
  }

  private final class UnaryNode extends LazyResult<NNResult> {
    private final UUID layer;
    private final LazyResult<NNResult>[] prevHead;

    @SafeVarargs
    private UnaryNode(final NNLayer<?> layer, final LazyResult<NNResult>... head) {
      assert(null != head);
      this.layer = layer.getId();
      this.prevHead = head;
    }

    @Override
    protected NNResult eval(final EvaluationContext ctx) {
      final NNResult input = (NNResult) this.prevHead[0].get(ctx);
      final NNResult output = DAGNetwork.this.byId.get(this.layer).eval(ctx, input);
      return output;
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.add("layer", DAGNetwork.this.byId.get(this.layer).getJson());
      json.add("prev", this.prevHead[0].toJson());
      return json;
    }
  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);

  private final java.util.LinkedHashMap<UUID, NNLayer<?>> byId = new java.util.LinkedHashMap<>();
  public final UUID inputHandle = UUID.randomUUID();
  private final InputNode inputNode = new InputNode();
  private LazyResult<NNResult> head = this.getInput().map(input->input[0]);

  private final java.util.HashMap<NNLayer<?>, NNLayer<?>> nextMap = new java.util.HashMap<>();
  private final java.util.HashMap<NNLayer<?>, NNLayer<?>> prevMap = new java.util.HashMap<>();

  public synchronized DAGNetwork add(final NNLayer<?> nextHead) {
    return add(nextHead, this.getHead());
  }

  @SafeVarargs
  public final DAGNetwork add(final NNLayer<?> nextHead, LazyResult<NNResult>... head) {
    this.byId.put(nextHead.getId(), nextHead);
    {
      final NNLayer<?> prevHead = getLayer(head[0]);
      this.prevMap.put(nextHead, prevHead);
      this.nextMap.put(prevHead, nextHead);
    }
    assert(null != this.getInput());
    final UnaryNode node = new UnaryNode(nextHead, head);
    setHead(node);
    return this;
  }

  public synchronized NNLayer<DAGNetwork> addLossComponent(final NNLayer<?> nextHead) {
    LazyResult<NNResult> idealNode = this.getInput().map(input->input[1]);
    this.byId.put(nextHead.getId(), nextHead);
    LazyResult<NNResult> head = this.getHead();
    final NNLayer<?> prevHead = getLayer(head);
    this.prevMap.put(nextHead, prevHead);
    this.nextMap.put(prevHead, nextHead);
    setHead(new BinaryNode(nextHead, head, idealNode));
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... array) {
    evaluationContext.cache.put(this.inputHandle, array);
    return (NNResult) getHead().get(evaluationContext);
  }

  @Override
  public NNLayer<DAGNetwork> evolve() {
    if (0 == this.byId.values().stream().filter(l -> {
      final NNLayer<?> evolve = l.evolve();
      if (null != evolve && evolve != l)
        throw new RuntimeException("Not implemented: Substitution via evolution in DAGNetwork");
      return null != evolve;
    }).count())
      return null;
    else
      return this;
  }

  @Override
  public DAGNetwork freeze() {
    this.byId.values().forEach(l -> l.freeze());
    return super.freeze();
  }

  public NNLayer<?> get(final int i) {
    return this.byId.get(i);
  }

  @Override
  public NNLayer<?> getChild(final UUID id) {
    if (this.id.equals(id))
      return this;
    if (this.byId.containsKey(id))
      return this.byId.get(id);
    return this.byId.values().stream().map(x -> x.getChild(id)).findAny().orElse(null);
  }

  @Override
  public List<NNLayer<?>> getChildren() {
    return this.byId.values().stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId())).collect(Collectors.toList());
  }

  public LazyResult<NNResult> getHead() {
    return this.head;
  }

  public NNLayer<?> getHeadLayer() {
    return getLayer(this.getHead());
  }

  public NNLayer<?> getLayer(LazyResult<NNResult> head) {
    if (head instanceof UnaryNode)
      return DAGNetwork.this.byId.get(((UnaryNode) head).layer);
    else if (head instanceof BinaryNode)
      return DAGNetwork.this.byId.get(((BinaryNode) head).layer);
    else
      return null;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("root", getHead().toJson());
    // for(NNLayer c : getChildren()){
    // json.add(c.getId(), c.getJson());
    // }
    return json;
  }

  private void permutate_back(final NNLayer<?> permutationLayer, final List<Tuple2<Integer, Integer>> permute) {
    final NNLayer<?> prev = this.prevMap.get(permutationLayer);
    final List<Tuple2<Integer, Integer>> passback = prev.permuteOutput(permute);
    if (null != passback) {
      permutate_back(prev, passback);
    }
  }

  private void permutate_forward(final NNLayer<?> permutationLayer, final List<Tuple2<Integer, Integer>> permute) {
    final NNLayer<?> next = this.nextMap.get(permutationLayer);
    final List<Tuple2<Integer, Integer>> passforward = next.permuteInput(permute);
    if (null != passforward) {
      permutate_forward(next, passforward);
    }
  }

  public void permute(final UUID id, final List<Tuple2<Integer, Integer>> permute) {
    final NNLayer<?> permutationLayer = this.byId.get(id);
    permutate_back(permutationLayer, permute);
    permutate_forward(permutationLayer, permute);

  }

  public void setHead(final LazyResult<NNResult> head) {
    this.head = head;
  }

  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l -> l.state().stream()).distinct().collect(Collectors.toList());
  }

  public LazyResult<NNResult[]> getInput() {
    return inputNode;
  }

}
