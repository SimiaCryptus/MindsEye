# PipelineNetwork
## ConvolutionNetworkTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.network.PipelineNetwork",
      "id": "a864e734-2f23-44db-97c1-504000002136",
      "isFrozen": false,
      "name": "PipelineNetwork/a864e734-2f23-44db-97c1-504000002136",
      "inputs": [
        "3e389b47-0413-40ba-bb7b-2742724b1d7e",
        "6a15ddad-0b1b-46bc-b148-622a3e49d4e6"
      ],
      "nodes": {
        "76a24ef1-810d-40e0-8c9f-ae6d15273b0b": "a864e734-2f23-44db-97c1-504000002137",
        "f388751c-9234-4a10-b5b9-86092c8ecfd7": "a864e734-2f23-44db-97c1-504000002138",
        "06560d54-b85c-4d24-8205-128c1158dcd5": "a864e734-2f23-44db-97c1-504000002139",
        "3b2a0da3-b275-48ac-ac71-580d7e962681": "a864e734-2f23-44db-97c1-50400000213a",
        "536214b0-f1a3-4b2e-9388-a803c0323a24": "a864e734-2f23-44db-97c1-50400000213c",
        "5f07f2b1-85d0-46b7-8375-bf70abe38573": "a864e734-2f23-44db-97c1-50400000213b"
      },
      "layers": {
        "a864e734-2f23-44db-97c1-504000002137": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
          "id": "a864e734-2f23-44db-97c1-504000002137",
          "isFrozen": false,
          "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000002137",
          "filter": {
            "dimensions": [
              3,
              3,
              21
            ],
            "data": [
              1.14,
              1.884,
              -0.916,
              1.46,
              0.932,
              0.956,
              0.172,
              1.608,
              -1.152,
              -0.56,
              -0.484,
              -1.0,
              0.848,
              0.42,
              -0.404,
              1.728,
              0.548,
              -0.8,
              -1.072,
              -1.12,
              1.076,
              -1.468,
              0.82,
              1.34,
              -0.684,
              1.476,
              1.908,
              -1.02,
              1.276,
              1.796,
              -1.6,
              1.012,
              0.968,
              -0.208,
              1.152,
              -0.644,
              1.288,
              -1.028,
              0.052,
              -1.728,
              1.452,
              0.264,
              1.552,
              0.056,
              -1.856,
              -0.352,
              -0.5,
              1.508,
        
```
...[skipping 2680 bytes](etc/1.txt)...
```
    rozen": false,
          "name": "ImgBandBiasLayer/a864e734-2f23-44db-97c1-504000002138",
          "bias": [
            0.0,
            0.0,
            0.0
          ]
        },
        "a864e734-2f23-44db-97c1-504000002139": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer",
          "id": "a864e734-2f23-44db-97c1-504000002139",
          "isFrozen": false,
          "name": "ActivationLayer/a864e734-2f23-44db-97c1-504000002139",
          "mode": 1
        },
        "a864e734-2f23-44db-97c1-50400000213a": {
          "class": "com.simiacryptus.mindseye.layers.java.ImgCropLayer",
          "id": "a864e734-2f23-44db-97c1-50400000213a",
          "isFrozen": false,
          "name": "ImgCropLayer/a864e734-2f23-44db-97c1-50400000213a",
          "sizeX": 4,
          "sizeY": 4
        },
        "a864e734-2f23-44db-97c1-50400000213c": {
          "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
          "id": "a864e734-2f23-44db-97c1-50400000213c",
          "isFrozen": false,
          "name": "MeanSqLossLayer/a864e734-2f23-44db-97c1-50400000213c"
        },
        "a864e734-2f23-44db-97c1-50400000213b": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "a864e734-2f23-44db-97c1-50400000213b",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/a864e734-2f23-44db-97c1-50400000213b",
          "power": 0.5
        }
      },
      "links": {
        "76a24ef1-810d-40e0-8c9f-ae6d15273b0b": [
          "6a15ddad-0b1b-46bc-b148-622a3e49d4e6"
        ],
        "f388751c-9234-4a10-b5b9-86092c8ecfd7": [
          "76a24ef1-810d-40e0-8c9f-ae6d15273b0b"
        ],
        "06560d54-b85c-4d24-8205-128c1158dcd5": [
          "f388751c-9234-4a10-b5b9-86092c8ecfd7"
        ],
        "3b2a0da3-b275-48ac-ac71-580d7e962681": [
          "06560d54-b85c-4d24-8205-128c1158dcd5"
        ],
        "536214b0-f1a3-4b2e-9388-a803c0323a24": [
          "3b2a0da3-b275-48ac-ac71-580d7e962681",
          "3e389b47-0413-40ba-bb7b-2742724b1d7e"
        ],
        "5f07f2b1-85d0-46b7-8375-bf70abe38573": [
          "536214b0-f1a3-4b2e-9388-a803c0323a24"
        ]
      },
      "labels": {},
      "head": "5f07f2b1-85d0-46b7-8375-bf70abe38573"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.24 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.01 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.144, -1.776, 1.172 ], [ -0.252, -0.276, 1.272 ], [ -1.52, -1.128, -0.068 ], [ -1.596, -0.472, -1.064 ] ],
    	[ [ 0.008, 1.772, -1.168 ], [ -1.152, -1.152, 0.272 ], [ -1.2, -0.368, 1.14 ], [ 1.252, 1.34, -0.472 ] ],
    	[ [ -1.004, -0.256, -0.544 ], [ 0.192, -2.0, -1.74 ], [ -0.164, -1.66, -0.62 ], [ -0.988, -1.652, -0.384 ] ],
    	[ [ -0.324, -1.512, -1.54 ], [ -1.164, 0.096, -1.556 ], [ 1.648, -1.832, -1.16 ], [ -0.152, 0.228, 0.268 ] ]
    ],
    [
    	[ [ 0.232, 1.096, -0.076, -0.44, -1.536, 0.076, 0.976 ], [ 1.636, -1.356, 0.328, 0.168, -1.864, -1.028, -1.772 ], [ 1.24, -1.38, 0.932, -0.356, 0.68, 0.676, -0.136 ], [ -0.772, -1.172, 0.948, 1.22, -1.816, -1.488, 1.872 ], [ -1.488, 1.892, -1.688, 0.26, -0.412, -1.264, 1.504 ] ],
    	[ [ -1.348, -0.004, 1.012, -1.012, 1.328, 0.044, 1.524 ], [ 1.588, -0.756, -0.596, -1.976, 0.888, 1.74, -0.964 ], [ 1.192, -1.488, -1.748, 1.428, -0.204, -1.476, -1.888 ], [ 0.664, 0.196, 0.656, -1.944, 0.176, -1.984, 1.424 ], [ 1.548, -0.06, 1.84, 1.04, 1.076, -1.828, -1.36 ] ],
    	[ [ -1.86, 1.476, 0.796, 1.424, -1.484, 0.548, -0.984 ], [ -0.62, 1.252, 1.54, 1.488, 1.468, -1.344, -1.884 ], [ 0.372, 1.988, 0.528, -1.312, -0.66, 1.492, 0.328 ], [ 1.756, 0.948, -0.812, 0.876, 1.34, -1.576, 0.08 ], [ 1.316, 0.216, -1.832, 0.564, 1.756, -0.56, -1.524 ] ],
    	[ [ 0.684, 1.052, 1.276, 1.668, -1.368, -0.288, 0.152 ], [ 0.032, -0.12, 1.656, 1.224, -0.26, 0.012, 1.532 ], [ -0.4, -0.168, -0.412, 0.24, 0.356, 1.068, -0.872 ], [ 1.404, 1.588, -1.976, 0.924, 1.624, -0.104, 1.104 ], [ 1.544, 1.42, -0.688, -1.224, -1.28, 1.136, -1.352 ] ],
    	[ [ 0.964, -0.144, 1.54, -1.888, -1.288, 1.52, -0.08 ], [ -1.188, 1.872, -1.272, 0.724, 1.08, 1.872, 0.588 ], [ 0.716, 1.384, 1.756, 0.512, -0.68, 0.896, -0.88 ], [ 1.336, 0.72, -1.988, 1.744, 0.804, -0.84, -0.344 ], [ 0.384, -0.54, -1.104, 1.924, -0.248, 0.696, 1.764 ] ]
    ]]
    --------------------
    Output: 
    [ 8.617157724995248 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.05 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.2284e-19 +- 3.8566e-18 [0.0000e+00 - 5.5511e-17] (2240#), relativeTol=1.7870e-17 +- 2.8049e-16 [0.0000e+00 - 8.9686e-15] (2233#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 1.35 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.144, -1.776, 1.172 ], [ -0.252, -0.276, 1.272 ], [ -1.52, -1.128, -0.068 ], [ -1.596, -0.472, -1.064 ] ],
    	[ [ 0.008, 1.772, -1.168 ], [ -1.152, -1.152, 0.272 ], [ -1.2, -0.368, 1.14 ], [ 1.252, 1.34, -0.472 ] ],
    	[ [ -1.004, -0.256, -0.544 ], [ 0.192, -2.0, -1.74 ], [ -0.164, -1.66, -0.62 ], [ -0.988, -1.652, -0.384 ] ],
    	[ [ -0.324, -1.512, -1.54 ], [ -1.164, 0.096, -1.556 ], [ 1.648, -1.832, -1.16 ], [ -0.152, 0.228, 0.268 ] ]
    ],
    [
    	[ [ 0.232, 1.096, -0.076, -0.44, -1.536, 0.076, 0.976 ], [ 1.636, -1.356, 0.328, 0.168, -1.864, -1.028, -1.772 ], [ 1.24, -1.38, 0.932, -0.356, 0.68, 0.676, -0.136 ], [ -0.772, -1.172, 0.948, 1.22, -1.816, -1.488, 1.872 ], [ -1.488, 1.892, -1.688, 0.26, -0.412, -1.264, 1.504 ] ],
    	[ [ -1.348, -0.004, 1.012, -1.012, 1.328, 0.044, 1.524 ], [ 1.588, -0.756, -0.596, -1.976, 0.888, 1.74, -0.964 ], [ 1.192, -1.488, -1.748, 1.428, -0.204, -1.476, -1.888 ], [ 0.664, 0.196, 0.656, -1.944, 0.176, -1.984, 1.424 ], [ 1.548, -0.06, 1.84, 1.04, 1.076, -1.828, -1.36 ] ],
    	[ [ -1.86, 1.476, 0.796, 1.424, -1.484, 0.548, -0.984 ], [ -0.62, 1.252, 1.54, 1.488, 1.468, -1.344, -1.884 ], [ 0.372, 1.988, 0.528, -1.312, -0.66, 1.492, 0.328 ], [ 1.756, 0.948, -0.812, 0.876, 1.34, -1.576, 0.08 ], [ 1.316, 0.216, -1.832, 0.564, 1.756, -0.56, -1.524 ] ],
    	[ [ 0.684, 1.052, 1.276, 1.668, -1.368, -0.288, 0.152 ], [ 0.032, -0.12, 1.656, 1.224, -0.26, 0.012, 1.532 ], [ -0.4, -0.168, -0.412, 0.24, 0.356, 1.068, -0.872 ], [ 1.404, 1.588, -1.976, 0.924, 1.624, -0.104, 1.104 ], [ 1.544, 1.42, -0.688, -1.224, -1.28, 1.136, -1.352 ] ],
    	[ [ 0.964, -0.144, 1.54, -1.888, -1.288, 1.52, -0.08 ], [ -1.188, 1.872, -1.272, 0.724, 1.08, 1.872, 0.588 ], [ 0.716, 1.384, 1.756, 0.512, -0.68, 0.896, -0.88 ], [ 1.336, 0.72, -1.988, 1.744, 0.804, -0.84, -0.344 ], [ 0.384, -0.54, -1.104, 1.924, -0.248, 0.696, 1.764 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.20007992758649987, negative=34, min=0.268, max=0.268, mean=-0.4815000000000001, count=48.0, positive=14, stdDev=0.9992369171856422, zeros=0},
    {meanExponent=-0.105
```
...[skipping 5885 bytes](etc/2.txt)...
```
    asured Gradient: [ [ 0.11761921117070528 ], [ 0.05885676566563802 ], [ 0.0716088377217261 ], [ 0.09464024170569019 ], [ 0.029382445045911254 ], [ 0.03363976277981351 ], [ 0.01062085686953651 ], [ 5.766349531199921E-4 ], ... ]
    Measured Statistics: {meanExponent=-1.3128703294683275, negative=77, min=-0.07960183785016284, max=-0.07960183785016284, mean=0.023764569100682705, count=189.0, positive=112, stdDev=0.09859772251902126, zeros=0}
    Gradient Error: [ [ 2.0239636680380313E-6 ], [ 1.583769380836264E-6 ], [ 1.2073375818544285E-6 ], [ 1.4986275077844002E-6 ], [ 1.11738002981851E-6 ], [ 1.0908900642606079E-6 ], [ 8.128147298905319E-7 ], [ 1.1605923630355177E-6 ], ... ]
    Error Statistics: {meanExponent=-5.89918545561361, negative=0, min=1.4637516668331019E-6, max=1.4637516668331019E-6, mean=1.3389057038247071E-6, count=189.0, positive=189, stdDev=4.233386263317472E-7, zeros=0}
    Learning Gradient for weight set 1
    Weights: [ 0.0, 0.0, 0.0 ]
    Implemented Gradient: [ [ 0.18727926904702094 ], [ 0.1832914498886237 ], [ 0.2685645012570479 ] ]
    Implemented Statistics: {meanExponent=-0.678439826824258, negative=0, min=0.2685645012570479, max=0.2685645012570479, mean=0.21304507339756418, count=3.0, positive=3, stdDev=0.03929190610285474, zeros=0}
    Measured Gradient: [ [ 0.18728027438186245 ], [ 0.18329222202240203 ], [ 0.26856529158791886 ] ]
    Measured Statistics: {meanExponent=-0.6784380138653993, negative=0, min=0.26856529158791886, max=0.26856529158791886, mean=0.21304592933072777, count=3.0, positive=3, stdDev=0.039291863699663074, zeros=0}
    Gradient Error: [ [ 1.005334841513994E-6 ], [ 7.721337783317406E-7 ], [ 7.903308709700241E-7 ] ]
    Error Statistics: {meanExponent=-6.070729255999642, negative=0, min=7.903308709700241E-7, max=7.903308709700241E-7, mean=8.559331636052528E-7, count=3.0, positive=3, stdDev=1.0590382292936666E-7, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2175e-06 +- 6.7428e-07 [1.4133e-08 - 3.4221e-06] (415#)
    relativeTol: 5.7858e-05 +- 2.6816e-04 [5.7941e-07 - 3.1157e-03] (415#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2175e-06 +- 6.7428e-07 [1.4133e-08 - 3.4221e-06] (415#), relativeTol=5.7858e-05 +- 2.6816e-04 [5.7941e-07 - 3.1157e-03] (415#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.73 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 30.9051 +- 25.3875 [22.4221 - 280.3736]
    Learning performance: 12.0900 +- 2.9979 [9.4300 - 34.9156]
    
```

