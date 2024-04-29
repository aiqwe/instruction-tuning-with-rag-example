echo -e "\033[032m 🚀 requirements.txt를 설치합니다 \033[0m"
pip install selenium openai colorama datasets accelerate==0.27.2 flash-attn peft trl transformers python-dotenv huggingface_hub

echo -e "\033[032m 🚀 SSH를 추가합니다... \033[0m"
ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa <<<y >/dev/null 2>&1

echo -e "\033[033m 🚀 아래 붉은 글씨를 복사하여 Github SSH Key에 추가하세요 \033[0m"
echo -e "\033[033m 👉 Github에서 우측상단 프로필 클릭 -> Settings -> SSH and GPG Keys -> SSH Keys / New SSH Key \033[0m"
echo -e "👉 참고 URL: \033[034m https://docs.github.com/ko/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent \033[0m"
echo -e "\033[031m $(cat ~/.ssh/id_rsa.pub) \033[0m"

echo "SSH Key 등록이 완료되셨나요? (y or Y / N or n)"
echo "N 또는 n을 입력하면 설치를 종료합니다."

while true; do
  read ssh_register_yn
  if [ "$ssh_register_yn" == "Y" ] || [ "$ssh_register_yn" == "y" ]; then
    break
  elif [ "$ssh_register_yn" == "N" ] || [ "$ssh_register_yn" == "n" ]; then
    exit 0
  else
    ssh_register_yn=2
    echo "Y, y, N, n 중 하나로 입력해주세요."
  fi
done

echo -e "\032[033m 🚀 gpt_with_tuning Repo에서 클론합니다. \033[0m"
git clone git@github.com:aiqwe/gpt_with_tuning.git

echo -e "\032[033m 🚀 원활한 import를 위해 PYTHONPATH 환경변수를 추가합니다. \033[0m"
export PYTHONPATH=$PYTHONPATH;/content/gpt_with_tuning/tuning

