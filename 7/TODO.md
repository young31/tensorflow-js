## Versions
- node: 14.15.3
- npm: 6.14.9
- tfjs: 2.7

## napi-v6
- 파일 하나를 옮기면 해결 가능
- move D:\TFJS\node_modules\@tensorflow\tfjs-node\deps\lib\tensorflow.dll to D:\TFJS\node_modules\@tensorflow\tfjs-node\lib\napi-v6\
- [관련이슈](https://github.com/tensorflow/tfjs/issues/4116)


## file path
- 파일 경로로 바로 접근은 왜인지 안됨
- 아래처럼 local url을 처리하고 하면 됨
- const handler = tfn.io.fileSystem("./path/to/your/model.json");
- [관련이슈](https://stackoverflow.com/questions/53639919/load-tensorflow-js-model-from-local-file-system-in-javascript)