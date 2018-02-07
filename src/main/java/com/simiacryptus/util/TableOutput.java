/*
 * Copyright (c) 2018 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.util;

import com.simiacryptus.util.data.DoubleStatistics;

import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

/**
 * The type Table output.
 */
public class TableOutput {
  
  /**
   * The Rows.
   */
  public final List<Map<String, Object>> rows = new ArrayList<>();
  /**
   * The Schema.
   */
  public final Map<String, Class<?>> schema = new LinkedHashMap<>();
  
  /**
   * Create table output.
   *
   * @param rows the rows
   * @return the table output
   */
  @javax.annotation.Nonnull
  public static TableOutput create(@javax.annotation.Nonnull final Map<String, Object>... rows) {
    @javax.annotation.Nonnull final TableOutput table = new TableOutput();
    Arrays.stream(rows).forEach(table::putRow);
    return table;
    
  }
  
  /**
   * Calc number stats table output.
   *
   * @return the table output
   */
  @javax.annotation.Nonnull
  public TableOutput calcNumberStats() {
    @javax.annotation.Nonnull final TableOutput tableOutput = new TableOutput();
    schema.entrySet().stream().filter(x -> Number.class.isAssignableFrom(x.getValue())).map(col -> {
      final String key = col.getKey();
      final DoubleStatistics stats = rows.stream().filter(x -> x.containsKey(key)).map(x -> (Number) x.get(key)).collect(DoubleStatistics.NUMBERS);
      @javax.annotation.Nonnull final LinkedHashMap<String, Object> row = new LinkedHashMap<>();
      row.put("field", key);
      row.put("sum", stats.getSum());
      row.put("avg", stats.getAverage());
      row.put("stddev", stats.getStandardDeviation());
      row.put("nulls", rows.size() - stats.getCount());
      return row;
    }).sorted(Comparator.comparing(x -> x.get("field").toString()))
          .forEach(row -> tableOutput.putRow(row));
    return tableOutput;
  }
  
  /**
   * Clear.
   */
  public void clear() {
    schema.clear();
    rows.clear();
  }
  
  /**
   * Put row.
   *
   * @param properties the properties
   */
  public void putRow(@javax.annotation.Nonnull final Map<String, Object> properties) {
    for (@javax.annotation.Nonnull final Entry<String, Object> prop : properties.entrySet()) {
      final String propKey = prop.getKey();
      final Class<?> propClass = prop.getValue().getClass();
      if (!propClass.equals(schema.getOrDefault(propKey, propClass))) {
        throw new RuntimeException("Schema mismatch for " + propKey);
      }
      schema.putIfAbsent(propKey, propClass);
    }
    rows.add(new HashMap<>(properties));
  }
  
  /**
   * To csv string.
   *
   * @param sortCols the sort cols
   * @return the string
   */
  public String toCSV(final boolean sortCols) {
    try (@javax.annotation.Nonnull ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {
      try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(buffer)) {
        @javax.annotation.Nonnull final Collection<String> keys = sortCols ? new TreeSet<String>(schema.keySet()) : schema.keySet();
        final String formatString = keys.stream()
                                        .map(k -> {
                                          switch (schema.get(k).getSimpleName()) {
                                            case "String":
                                              return "%-" + rows.stream().mapToInt(x -> x.getOrDefault(k, "").toString().length()).max().getAsInt() + "s";
                                            case "Integer":
                                              return "%6d";
                                            case "Double":
                                              return "%.4f";
                                            default:
                                              return "%s";
                                          }
                                        }).collect(Collectors.joining(","));
        printStream.println(keys.stream().collect(Collectors.joining(",")).trim());
        for (@javax.annotation.Nonnull final Map<String, Object> row : rows) {
          printStream.println(String.format(formatString, keys.stream().map(k -> row.get(k)).toArray()));
        }
      }
      return buffer.toString();
    } catch (@javax.annotation.Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * To html table string.
   *
   * @return the string
   */
  public String toHtmlTable() {
    return toHtmlTable(false);
  }
  
  /**
   * To html table string.
   *
   * @param sortCols the sort cols
   * @return the string
   */
  public String toHtmlTable(final boolean sortCols) {
    try (@javax.annotation.Nonnull ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {
      try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(buffer)) {
        @javax.annotation.Nonnull final Collection<String> keys = sortCols ? new TreeSet<String>(schema.keySet()) : schema.keySet();
        final String formatString = keys.stream()
                                        .map(k -> {
                                          switch (schema.get(k).getSimpleName()) {
                                            case "String":
                                              return "%-" + rows.stream().mapToInt(x -> x.getOrDefault(k, "").toString().length()).max().getAsInt() + "s";
                                            case "Integer":
                                              return "%6d";
                                            case "Double":
                                              return "%.4f";
                                            default:
                                              return "%s";
                                          }
                                        }).map(s -> "<td>" + s + "</td>").collect(Collectors.joining(""));
        printStream.print("<table border=1>");
        printStream.print("<tr>");
        printStream.println(keys.stream().map(s -> "<th>" + s + "</th>").collect(Collectors.joining("")).trim());
        printStream.print("</tr>");
        for (@javax.annotation.Nonnull final Map<String, Object> row : rows) {
          printStream.print("<tr>");
          printStream.println(String.format(formatString, keys.stream().map(k -> row.get(k)).toArray()));
          printStream.print("</tr>");
        }
        printStream.print("</table>");
      }
      return buffer.toString();
    } catch (@javax.annotation.Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * To text table string.
   *
   * @return the string
   */
  public String toTextTable() {
    try (@javax.annotation.Nonnull ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {
      try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(buffer)) {
        final String formatString = schema.entrySet().stream()
                                          .map(e -> {
                                            switch (e.getValue().getSimpleName()) {
                                              case "String":
                                                return "%-" + rows.stream().mapToInt(x -> x.getOrDefault(e.getKey(), "").toString().length()).max().getAsInt() + "s";
                                              case "Integer":
                                                return "%6d";
                                              case "Double":
                                                return "%.4f";
                                              default:
                                                return "%s";
                                            }
                                          }).collect(Collectors.joining(" | "));
        printStream.println(schema.entrySet().stream().map(x -> x.getKey()).collect(Collectors.joining(" | ")).trim());
        printStream.println(schema.entrySet().stream().map(x -> x.getKey()).map(x -> {
          @javax.annotation.Nonnull final char[] t = new char[x.length()];
          Arrays.fill(t, '-');
          return new String(t);
        }).collect(Collectors.joining(" | ")).trim());
        for (@javax.annotation.Nonnull final Map<String, Object> row : rows) {
          printStream.println(String.format(formatString, schema.entrySet().stream().map(e -> row.get(e.getKey())).toArray()));
        }
      }
      return buffer.toString();
    } catch (@javax.annotation.Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Write projector data.
   *
   * @param path    the path
   * @param baseUrl the base url
   * @throws IOException the io exception
   */
  public void writeProjectorData(@javax.annotation.Nonnull final File path, final URL baseUrl) throws IOException {
    path.mkdirs();
    try (@javax.annotation.Nonnull FileOutputStream file = new FileOutputStream(new File(path, "data.tsv"))) {
      try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(file)) {
        printStream.println(toTextTable());
      }
    }
    final List<Entry<String, Class<?>>> scalarCols = schema.entrySet().stream()
                                                           .filter(e -> Number.class.isAssignableFrom(e.getValue()))
                                                           .collect(Collectors.toList());
    try (@javax.annotation.Nonnull FileOutputStream file = new FileOutputStream(new File(path, "tensors.tsv"))) {
      try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(file)) {
        for (@javax.annotation.Nonnull final Map<String, Object> row : rows) {
          printStream.println(scalarCols.stream()
                                        .map(e -> ((Number) row.getOrDefault(e.getKey(), 0)).doubleValue())
                                        .map(x -> x.toString()).collect(Collectors.joining("\t")));
        }
      }
    }
    final List<Entry<String, Class<?>>> metadataCols = schema.entrySet().stream()
                                                             .filter(e -> String.class.isAssignableFrom(e.getValue()))
                                                             .collect(Collectors.toList());
    try (@javax.annotation.Nonnull FileOutputStream file = new FileOutputStream(new File(path, "metadata.tsv"))) {
      try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(file)) {
        if (1 < metadataCols.size()) {
          printStream.println(metadataCols.stream().map(e -> e.getKey()).collect(Collectors.joining("\t")));
        }
        for (@javax.annotation.Nonnull final Map<String, Object> row : rows) {
          printStream.println(metadataCols.stream()
                                          .map(e -> ((String) row.getOrDefault(e.getKey(), "")))
                                          .collect(Collectors.joining("\t")));
        }
      }
    }
    final List<Entry<String, Class<?>>> urlCols = schema.entrySet().stream()
                                                        .filter(e -> URL.class.isAssignableFrom(e.getValue()))
                                                        .collect(Collectors.toList());
    try (@javax.annotation.Nonnull FileOutputStream file = new FileOutputStream(new File(path, "bookmarks.txt"))) {
      try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(file)) {
        for (@javax.annotation.Nonnull final Map<String, Object> row : rows) {
          printStream.println(urlCols.stream()
                                     .map(e -> row.get(e.getKey()).toString())
                                     .collect(Collectors.joining("\t")));
        }
      }
    }
    try (@javax.annotation.Nonnull FileOutputStream file = new FileOutputStream(new File(path, "config.json"))) {
      try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(file)) {
        printStream.println("{\n" +
                              "  \"embeddings\": [\n" +
                              "    {\n" +
                              "      \"tensorName\": \"" + path.getName() + "\",\n" +
                              "      \"tensorShape\": [\n" +
                              "        " + rows.size() + ",\n" +
                              "        " + scalarCols.size() + "\n" +
                              "      ],\n" +
                              "      \"tensorPath\": \"" + new URL(baseUrl, "tensors.tsv") +
                              (0 == metadataCols.size() ? "" : "\",\n      \"metadataPath\": \"" + new URL(baseUrl, "metadata.tsv")) +
                              "\"\n" +
                              "    }\n" +
                              "  ]\n" +
                              "}");
      }
    }
    if (0 < urlCols.size()) {
      try (@javax.annotation.Nonnull FileOutputStream file = new FileOutputStream(new File(path, "config_withLinks.json"))) {
        try (@javax.annotation.Nonnull PrintStream printStream = new PrintStream(file)) {
          printStream.println("{\n" +
                                "  \"embeddings\": [\n" +
                                "    {\n" +
                                "      \"tensorName\": \"" + path.getName() + "\",\n" +
                                "      \"tensorShape\": [\n" +
                                "        " + rows.size() + ",\n" +
                                "        " + scalarCols.size() + "\n" +
                                "      ],\n" +
                                "      \"tensorPath\": \"" + new URL(baseUrl, "tensors.tsv") +
                                (0 == metadataCols.size() ? "" : "\",\n      \"metadataPath\": \"" + new URL(baseUrl, "metadata.tsv")) +
                                "\",\n      \"bookmarksPath\": \"" + new URL(baseUrl, "bookmarks.txt") +
                                "\"\n" +
                                "    }\n" +
                                "  ]\n" +
                                "}");
        }
      }
    }
    
  }
}
