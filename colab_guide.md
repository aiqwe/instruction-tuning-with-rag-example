# Colab 실행 가이드
GPU 사용을 위해 Colab에서 실행해보겠습니다.

---
1. google Colab에 접속하기  
[https://colab.google//](https://colab.google//)  
Colab에서 새 노트를 작성합니다.  

2. google Drive 마운트하기
현재 생성한 Colab 노트북은 임시 파일입니다. 드라이브에 마운트하여 파일을 저장할 수 있도록 해보겠습니다.  
![](assets/colab_drive_mount.png)


2. github의 레포를 clone하기  
![process](assets/colab1_resize.png)  
github의 https 주소를 복사한뒤, colab에서 실행합니다.  

```
!git clone https://github.com/aiqwe/instruction-tuning-with-rag-example.git
```

<br>
<br>

2. 구글 드라이브에 코드 업로드하고 코랩 실행하기  
![process](assets/colab2_resize.png)  
구글 드라이브에 코드 파일을 업로드하고, Jupyter Notebook 파일을 Colab과 연결하여 실행시킵니다.  
위 사진처럼 `train.ipynb` 파일을 마우스 우클릭하고 Google Colaboratory로 실행시킵니다.  

<br>
<br>
  
3. Colab에서 Secret, GPU 설정하기  
![process](assets/colab3_resize.png)  
왼쪽 세로 네게이션 바를 보면 위 사진과 같이 열쇠 그림이 있습니다. 여기서 Secret을 설정할 수 있습니다.  
사진처럼 오른쪽 상단 빨간색 박스 안에는 사용하고 있는 GPU가 표시됩니다. 여기서 L4 또는 A100을 사용합니다.  
예제 코드는 `torch.bfloat16` 타입을 사용하기 때문에 Ampere 7 이전의 GPU에서는 지원되지 않습니다.  
+ Secret을 설정하는 방법은 [여기](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75)를 참조해 주세요.
+ `utils.py`의 함수들은 아래처럼 `.env` 파일을 만들면 자동으로 환경변수를 읽습니다.(또는 시스템 환경변수를 읽습니다.)
```bash
# .env 파일
OPENAI_API_KEY=발급받은KEY
```
+ `train.ipynb` 파일의 huggingface토큰은 함수 인자에 직접 입력하거나 Colab Secret을 활용해주세요.(Colab을 가정하기 때문에 환경변수로 읽지 않습니다.)

<br>
<br>

5. `import`를 위한 `path` 지정하기  
![process](assets/colab4_resize.png)  
다시 왼쪽 세로 네게이션 바를 보면 위 사진과 같이 폴더 그림이 있습니다. 여기서 연결된 구글 드라이브로 탐색할 수 있습니다.  
`drive.mount('/content/drive')` 코드로 구글 드라이브를 연결하면 위 사진처럼 드라이브의 폴더들이 보입니다.  
`utils.py`, `prompts.py`, `similarity.py` 모듈을 `import`하기 위해 위 사진처럼 `sys.path`에 추가합니다.

<br>
<br>

+ `train.ipynb` 에서 huggingface 토큰을 지정해야합니다.  
Gemma 모델을 로드하기 위해서는 huggingface의 gemma 모델 사용 신청을 하고, huggingface 토큰을 발급받아야 합니다.  
토큰 발급은 huggingface의 [User Access Tokens](https://huggingface.co/docs/hub/security-tokens)를 참조하세요.  