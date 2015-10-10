package com.simiacryptus.mindseye.core.delta;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Stream;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.NDArray;

/**
 * Nonlinear Network Layer (aka Neural Network Layer)
 *
 * @author Andrew Charneski
 */
public abstract class NNLayer<T extends NNLayer<T>> implements java.io.Serializable {

  private static final class ConstNNResult extends NNResult {

    // public final String[] created = Util.currentStack();

    private ConstNNResult(final NDArray data) {
      super(data);
    }

    @Override
    public void accumulate(final DeltaSet buffer, final NDArray data) {
      // Do Nothing
    }

    @Override
    public boolean isAlive() {
      return false;
    }
  }

  /**
   * 
   */
  private static final long serialVersionUID = 8741041477497062122L;

  public static NNResult[] getConstResult(final NDArray... array) {
    return Stream.of(array).map(a -> new ConstNNResult(a)).toArray(i -> new NNResult[i]);
  }

  private boolean frozen = false;

  public final UUID id = Util.uuid();

  @Override
  public boolean equals(final Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    final NNLayer<?> other = (NNLayer<?>) obj;
    if (this.id == null) {
      if (other.id != null)
        return false;
    } else if (!this.id.equals(other.id))
      return false;
    return true;
  }

  public final NNResult eval(final NDArray... array) {
    return eval(NNLayer.getConstResult(array));
  }

  public abstract NNResult eval(NNResult... array);

  public NNLayer<?> evolve() {
    return null;
  }

  public T freeze() {
    return setFrozen(true);
  }

  public NNLayer<?> getChild(final UUID id) {
    if (this.id.equals(id))
      return this;
    return null;
  }

  public List<NNLayer<?>> getChildren() {
    return Arrays.asList(this);
  }

  public final UUID getId() {
    return this.id;
  }

  public JsonObject getJson() {
    final JsonObject json = new JsonObject();
    json.addProperty("class", getClass().getSimpleName());
    json.addProperty("id", getId().toString());
    return json;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + (this.id == null ? 0 : this.id.hashCode());
    return result;
  }

  public final boolean isFrozen() {
    return this.frozen;
  }

  @SuppressWarnings("unchecked")
  protected final T self() {
    return (T) this;
  }

  public final T setFrozen(final boolean frozen) {
    this.frozen = frozen;
    return self();
  }

  public abstract List<double[]> state();

  @Override
  public final String toString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(getJson());
  }

}
