# Report template for weekly meetings or other regular updates

## 0. 準備作業
- [Docker](https://www.docker.com/get-started)のインストール
- Docker が起動していることを確認
  - Windows/Mac: Docker Desktop アプリケーションを起動
  - Linux: `systemctl status docker` で Docker サービスが起動していることを確認
- [dev container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 拡張機能がvscodeにインストールしてある

> [!TIP]
> vscode以外のEditorを使用している場合:
>  - cursor
>  - antigravity
>    
> などの場合は，各Editorに対応するdev containerの拡張機能を使用してください．
## 1. このリポジトリの使い方
1. このテンプレートリポジトリを使う:
   - 右上の「Use this template」ボタンをクリック
   - 「Create a new repository」オプションを選択
   - リポジトリ名を入力し、「Create repository from template」ボタンをクリック
2. クローンする:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   ```
3. ディレクトリに移動する:
   ```bash
   cd <YOUR_REPOSITORY_NAME>
   ```
4. vscode で開く:
   ```bash
   code .
   ```
5. dev container を起動する:
   - vscode の左下に「><」のアイコンが表示されるのでクリック
   - 「Reopen in Container」を選択

> [!NOTE]
> 初回の起動時は，dev container のビルドに時間がかかる場合があります．初回の拡張機能のインストールや設定も時間がかかるため，しばらく待機が必要．


## 2. レポートの作成方法
1. `template`フォルダをコピーして，新しいフォルダを作成する:
   - フォルダ名は `YYYY.MM.DD` の形式にすることをおすすめ (例: `2024.06.15`)
   - 同様にファイル名も `data.tex` から `YYYY.MM.DD.tex` に変更することをおすすめ (例: `2024.06.15.tex`)
2. コピーした `.tex` ファイルを編集してレポート内容を記入
3. `cmd` + `c` (mac) または `ctrl` + `c` (windows/linux) でコンパイル
