import os


def add_readme(name, dict_, string="", end="<br><br>\n"):
    if string != "":
        string += "\n"

    if "Картинка" in dict_:
        string += f"**{name}:**\n<br>" \
                  f"![alt text]({dict_['Картинка']})"
    if "Таблица" in dict_:
        dict_["Колонки"] = dict_["Таблица"].columns
        dict_["Значения"] = dict_["Таблица"].values
        dict_['Индексы'] = dict_["Таблица"].index
        string += f"**{name}:**\n<br><table>" \
                  f"<tr><th>Индексы</th>" + "".join(
            map(lambda x: f"<th>{x}</th>", dict_["Колонки"])) + "</tr>"
        for row in range(len(dict_["Значения"])):
            string += f"<tr><th>{dict_['Индексы'][row]}</th>" + "".join(
                map(lambda x: f"<th>{x:.2f}</th>", dict_["Значения"][row])) + "</tr>"
        string += "</table>"
    if "Текст" in dict_:
        string += f"**{name}:**<br>\n{dict_['Текст']}"
    return string + end


def add_global_readme():
    files = os.listdir("./")
    all_dir = "# Мой путь в обучении:<br> \n"
    dir_data = []
    for file in files:
        if not ("." in file):
            if os.path.exists("./" + file + "/README.md"):
                with open(file + "/README.md", "r") as f:
                    link = f"[{file}]({file})"
                    dir_data.append([os.path.getmtime("./" + file), "**" + link + "** - *" + f.readline().replace("# ", "").replace("\n", "*<br>\n")])
    dir_data.sort(key=lambda x: x[0])
    all_dir += "".join(map(lambda x: x[1], dir_data))
    with open("README.md", "w") as readme:
        readme.write(all_dir)