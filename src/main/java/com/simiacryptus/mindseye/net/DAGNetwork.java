package com.simiacryptus.mindseye.net;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.core.NNLayer;
import com.simiacryptus.mindseye.core.NNResult;

/***
 * Builds a network NNLayer components, assumed to form a directed acyclic graph
 * with a single output. Supplied builder methods designed to build linear
 * sequence of units acting on the current output node.
 *
 * @author Andrew Charneski
 */
public class DAGNetwork extends NNLayer<DAGNetwork> {

  public interface DAGNode {
    NNResult get(EvaluationContext buildExeCtx);
  }

  public class EvaluationContext {

    public final Map<UUID, NNResult> cache = new HashMap<>();

  }

  private final class InnerNode extends LazyResult {
    private final UUID layer;
    private final DAGNode[] prevHead;

    @SafeVarargs
    private InnerNode(final NNLayer<?> layer, final DAGNode... head) {
      assert null != head;
      this.layer = layer.getId();
      this.prevHead = head;
    }

    @Override
    protected NNResult eval(final EvaluationContext ctx) {
      final NNResult[] in = java.util.Arrays.stream(this.prevHead).map(x -> ((LazyResult) x).get(ctx)).toArray(i -> new NNResult[i]);
      final NNResult output = DAGNetwork.this.byId.get(this.layer).eval(in);
      return output;
    }

    @Override
    public JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.add("layer", DAGNetwork.this.byId.get(this.layer).getJson());
      json.add("prev", ((LazyResult) this.prevHead[0]).toJson());
      return json;
    }
  }

  private final class InputNode extends LazyResult {
    public final UUID handle;

    private InputNode() {
      this(null);
    }

    public InputNode(final UUID handle) {
      super(handle);
      this.handle = handle;
    }

    @Override
    protected NNResult eval(final EvaluationContext t) {
      return t.cache.get(this.handle);
    }

    @Override
    public JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.addProperty("target", DAGNetwork.this.inputHandles.toString());
      return json;
    }
  }

  private abstract static class LazyResult implements DAGNode {

    public final UUID key;

    public LazyResult() {
      this(UUID.randomUUID());
    }

    protected LazyResult(final UUID key) {
      super();
      this.key = key;
    }

    protected abstract NNResult eval(EvaluationContext t);

    @Override
    public NNResult get(final EvaluationContext t) {
      return t.cache.computeIfAbsent(this.key, k -> eval(t));
    }

    public abstract JsonObject toJson();

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);

  /**
   * 
   */
  private static final long serialVersionUID = -5683282519002886564L;

  private final java.util.LinkedHashMap<UUID, NNLayer<?>> byId = new java.util.LinkedHashMap<>();
  public final List<UUID> inputHandles = new java.util.ArrayList<>(java.util.Arrays.asList(UUID.randomUUID(), UUID.randomUUID()));
  private LazyResult head = getInput().get(0);

  LazyResult inputNode = new InputNode();
  private final java.util.HashMap<NNLayer<?>, NNLayer<?>> nextMap = new java.util.HashMap<>();

  private final java.util.HashMap<NNLayer<?>, NNLayer<?>> prevMap = new java.util.HashMap<>();

  public synchronized DAGNetwork add(final NNLayer<?> nextHead) {
    return add(nextHead, getHead());
  }

  @SafeVarargs
  public final DAGNetwork add(final NNLayer<?> nextHead, final DAGNode... head) {
    this.byId.put(nextHead.getId(), nextHead);
    {
      // XXX: Prev/next linking only tracks first input node
      final NNLayer<?> prevHead = getLayer(head[0]);
      this.prevMap.put(nextHead, prevHead);
      this.nextMap.put(prevHead, nextHead);
    }
    assert null != getInput();
    final InnerNode node = new InnerNode(nextHead, head);
    setHead(node);
    return this;
  }

  public synchronized NNLayer<DAGNetwork> addLossComponent(final NNLayer<?> nextHead) {
    return add(nextHead, getHead(), getInput().get(1));
  }

  public EvaluationContext buildExeCtx(final NNResult... array) {
    final EvaluationContext evaluationContext = new EvaluationContext();
    for (int i = 0; i < array.length; i++) {
      evaluationContext.cache.put(this.inputHandles.get(i), array[i]);
    }
    return evaluationContext;
  }

  @Override
  public NNResult eval(final NNResult... array) {
    return ((LazyResult) getHead()).get(buildExeCtx(array));
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

  public DAGNode getHead() {
    return this.head;
  }

  public NNLayer<?> getHeadLayer() {
    return getLayer(getHead());
  }

  public List<LazyResult> getInput() {
    return com.google.common.collect.Lists.transform(this.inputHandles, h -> new InputNode(h));
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("root", ((LazyResult) getHead()).toJson());
    // for(NNLayer c : getChildren()){
    // json.add(c.getId(), c.getJson());
    // }
    return json;
  }

  public NNLayer<?> getLayer(final DAGNode head) {
    if (head instanceof InnerNode)
      return DAGNetwork.this.byId.get(((InnerNode) head).layer);
    else
      return null;
  }

  public void setHead(final DAGNode imageRMS) {
    this.head = (LazyResult) imageRMS;
  }

  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l -> l.state().stream()).distinct().collect(Collectors.toList());
  }

}
