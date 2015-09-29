package com.simiacryptus.mindseye.math;

import java.util.Arrays;

public class Coordinate {
  public static int[] add(final int[] a, final int[] b) {
    final int[] r = new int[Math.max(a.length, b.length)];
    for (int i = 0; i < r.length; i++) {
      r[i] = (a.length <= i ? 0 : a[i]) + (b.length <= i ? 0 : b[i]);
    }
    return r;
  }

  public final int[] coords;

  public final int index;

  public Coordinate(final int index, final int[] coords) {
    super();
    this.index = index;
    this.coords = coords;
  }

  @Override
  public boolean equals(final Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    final Coordinate other = (Coordinate) obj;
    if (!Arrays.equals(this.coords, other.coords))
      return false;
    return true;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(this.coords);
    return result;
  }

  @Override
  public String toString() {
    return Arrays.toString(this.coords) + "<" + this.index + ">";
  }

}
