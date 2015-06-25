package com.simiacryptus.mindseye;

import java.util.Arrays;

public class Coordinate {
  public final int[] coords;
  public final int index;
  
  public Coordinate(int index, int[] coords) {
    super();
    this.index = index;
    this.coords = coords;
  }
  
  @Override
  public String toString() {
    return Arrays.toString(coords);
  }
  
  public static int[] add(int[] a, int[] b) {
    int[] r = new int[a.length];
    for(int i=0;i<r.length;i++) r[i] = a[i] + b[i];
    return r; 
  }
  
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(coords);
    return result;
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (getClass() != obj.getClass()) return false;
    Coordinate other = (Coordinate) obj;
    if (!Arrays.equals(coords, other.coords)) return false;
    return true;
  }
  
}