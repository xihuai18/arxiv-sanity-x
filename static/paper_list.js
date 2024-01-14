'use strict';

const UTag = props => {
    const tag_name = props.tag;
    const turl = "/?rank=tags&tags=" + encodeURIComponent(tag_name);
    return (
        <div class='rel_utag'>
            <a href={turl}>
                {tag_name}
            </a>
        </div>
    )
}

const Paper = props => {
    const p = props.paper;
    const lst = props.tags;
    const tlst = lst.map((jtag, ix) => jtag.name);
    const ulst = p.utags;
    const renderlst = tlst.map(item => ulst.includes(item) ? `*${item}*` : item);
    const adder = () => props.addTag(p, renderlst);
    const subber = () => props.subTag(p, ulst);

    const utags = p.utags.map((utxt, ix) => <UTag key={ix} tag={utxt} />);
    const similar_url = "/?rank=pid&pid=" + encodeURIComponent(p.id);
    const inspect_url = "/inspect?pid=" + encodeURIComponent(p.id);
    const thumb_img = p.thumb_url === '' ? null : <div class='rel_img'><img src={p.thumb_url} /></div>;
    // if the user is logged in then we can show add/sub buttons
    let utag_controls = null;
    if (user) {
        utag_controls = (
            <div class='rel_utags'>
                <div>
                    <input
                        list="tags-list"
                        value={props.target_tag}
                        onChange={props.onInputTag}
                        placeholder="Add a tag"
                    />
                    <datalist id="tags-list">
                        {tlst.map(tag => (
                            <option key={tag} value={tag} />
                        ))}
                    </datalist>
                    <button onClick={adder}>Add Tag</button>
                </div>
                <div class="rel_utag rel_utag_sub" onClick={subber}>-</div>
                {utags}
            </div>
        )
    }

    return (
        <div class='rel_paper'>
            <div class="rel_score">{p.weight.toFixed(2)}</div>
            <div class='rel_title'><a href={'http://arxiv.org/abs/' + p.id}>{p.title}</a></div>
            <div class='rel_authors'>{p.authors}</div>
            <div class="rel_time">{p.time}</div>
            <div class='rel_tags'>{p.tags}</div>
            {utag_controls}
            {thumb_img}
            <div class='rel_abs'>{p.summary}</div>
            <div class='rel_more'><a href={similar_url}>similar</a></div>
            <div class='rel_inspect'><a href={inspect_url}>inspect</a></div>
        </div>
    )
}

const PaperList = props => {
    const lst = props.papers;
    const filtered_tags = props.tags.filter(tag => tag.name !== 'all');
    const plst = lst.map((jpaper, ix) => <PaperComponent key={ix} paper={jpaper} tags={filtered_tags} />);
    return (
        <div>
            <div id="paperList" class="rel_papers">
                {plst}
            </div>
        </div>
    )
}

class PaperComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = { key: props.key, paper: props.paper, tags: props.tags, target_tag: '' };
        this.addTag = this.addTag.bind(this);
        this.subTag = this.subTag.bind(this);
    }
    handleTagInputChange = (event) => {
        this.setState({ target_tag: event.target.value });
    };
    addTag() {
        const { paper, tags, target_tag } = this.state
        fetch("/add/" + paper.id + "/" + target_tag)
            .then(response => console.log(response.text()))
            .then(() => {
                paper.utags = [...paper.utags, target_tag];
                this.setState((prevState) => ({
                    ...prevState,
                    paper: paper,
                }));
            });
        this.setState({ target_tag: '' })
    }
    subTag(paper, utlst) {
        let tagname = prompt(`tag to subtract from this paper, paper tags:\n${utlst.join('\n')}`);
        fetch("/sub/" + paper.id + "/" + tagname)
            .then(response => console.log(response.text()))
            .then(() => {
                paper.utags = paper.utags.filter(tag => tag !== tagname);
                this.setState((prevState) => ({
                    ...prevState,
                    paper: paper,
                }));
            });
    }
    render() {
        return (
            <Paper key={this.state.key} paper={this.state.paper} tags={this.state.tags} onInputTag={this.handleTagInputChange} addTag={this.addTag} subTag={this.subTag} />
        );
    }
}

const Tag = props => {
    const t = props.tag;
    const turl = "/?rank=tags&tags=" + encodeURIComponent(t.name);
    const tag_class = 'rel_utag' + (t.name === 'all' ? ' rel_utag_all' : '');
    return (
        <div class={tag_class}>
            <a href={turl}>
                {t.n} {t.name}
            </a>
        </div>
    )
}



const TagList = props => {
    const lst = props.tags;
    const tlst = lst.map((jtag, ix) => <Tag key={ix} tag={jtag} />);
    const deleter = props.deleteTag;
    const renamer = props.renameTag;
    // show the #wordwrap element if the user clicks inspect
    const show_inspect = () => { document.getElementById("wordwrap").style.display = "block"; };
    const inspect_elt = words.length > 0 ? <div id="inspect_svm" onClick={show_inspect}>inspect</div> : null;
    return (
        <div>
            <div>
                <div class="rel_tag_rename" onClick={renamer}>✍︎</div>
                <div class="rel_tag_sub" onClick={deleter}>-</div>
            </div>
            <div id="tagList" class="rel_utags">
                {tlst}
            </div>
            {inspect_elt}
        </div>
    )
}


class TagListComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = { tags: props.tags };
        this.deleteTag = this.deleteTag.bind(this);
        this.renameTag = this.renameTag.bind(this);
    }
    deleteTag() {
        const lst = this.state.tags;
        const filtered_lst = lst.filter(tag => tag.name !== 'all');
        const tlst = filtered_lst.map((jtag, ix) => jtag.name);
        let tagname = prompt(`delete tag name:\n${tlst.join('\n')}`);
        if (tagname === null) {
            console.log("Tag deleting cancelled.");
            return;
        }
        fetch("/del/" + tagname)
            .then(response => {
                this.setState((prevState) => ({
                    tags: prevState.tags.filter(tag => tag.name !== tagname)
                }), () => {
                    console.log(response.text());
                });
            });
    }
    renameTag() {
        const lst = this.state.tags;
        const filtered_lst = lst.filter(tag => tag.name !== 'all');
        const tlst = filtered_lst.map((jtag, ix) => jtag.name);
        let oldTagName = prompt(`Enter tag name to rename:\n${tlst.join('\n')}`);
        if (oldTagName === null) {
            console.log("Tag renaming 1 cancelled.");
            return;
        }

        let newTagName = prompt(`Enter new tag name for \`${oldTagName}\`:\n`);
        if (newTagName === null) {
            console.log("Tag renaming 2 cancelled.");
            return;
        }
        fetch("/rename/" + oldTagName + "/" + newTagName)
            .then(response => {
                this.setState((prevState) => ({
                    tags: prevState.tags.map(tag => tag.name === oldTagName ? { ...tag, name: newTagName } : tag)
                }), () => {
                    console.log(response.text());
                });
            });
    }
    render() {
        return (
            <TagList tags={this.state.tags} deleteTag={this.deleteTag} renameTag={this.renameTag} />
        );
    }
}

const CombinedTag = props => {
    const t = props.comtag;
    const turl = "/?rank=tags&logic=and&tags=" + encodeURIComponent(t.name);
    const tag_class = 'rel_utag rel_utag_all'
    return (
        <div class={tag_class}>
            <a href={turl}>
                {t.name}
            </a>
        </div>
    )
}

const CombinedTagList = props => {
    const lst = props.combined_tags;
    const tlst = lst.map((jtag, ix) => <CombinedTag key={ix} comtag={jtag} />);
    const deleter = props.deleteTag;
    const adder = props.addTag;
    return (
        <div>
            <div>
                <div class="rel_tag_rename" onClick={adder}>+</div>
                <div class="rel_tag_sub" onClick={deleter}>-</div>
            </div>
            <div id="tagList" class="rel_utags">
                {tlst}
            </div>
        </div>
    )
}

class CombinedTagListComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = { combined_tags: props.combined_tags, tags: props.tags };
        this.deleteTag = this.deleteTag.bind(this);
        this.addTag = this.addTag.bind(this);
    }
    deleteTag() {
        const lst = this.state.combined_tags;
        const tlst = lst.map((jtag, ix) => jtag.name);
        let ctagname = prompt(`delete combined tag:\n${tlst.join('\n')}`);
        fetch("/del_ctag/" + ctagname)
            .then(response => {
                this.setState((prevState) => ({
                    combined_tags: prevState.combined_tags.filter(tag => tag.name !== ctagname)
                }), () => {
                    console.log(response.text());
                });
            });
    }
    addTag() {
        const lst = this.state.tags;
        const tlst = lst.map((jtag, ix) => jtag.name);
        let ctagname = prompt(`Register a combination of tags (seperated by ', '), existings tags:\n${tlst.join('\n')}`);
        if (ctagname) {
            fetch("/add_ctag/" + ctagname)
                .then(response => response.text())
                .then(text => {
                    if (text.includes('ok')) {
                        this.setState((prevState) => ({
                            combined_tags: [...prevState.combined_tags, { name: ctagname }]
                        }), () => {
                            console.log(text);
                        });
                    } else {
                        console.log("Response does not contain 'ok':", text);
                    }
                });
        } else {
            console.log("Tag addition cancelled.");
        }

    }
    render() {
        return (
            <CombinedTagList combined_tags={this.state.combined_tags} deleteTag={this.deleteTag} addTag={this.addTag} />
        );
    }
}


const Key = props => {
    const k = props.jkey;
    const kurl = `/?q=${encodeURIComponent(k.name)}&rank=search`;
    const key_class = 'rel_ukey' + (k.name === 'Artificial general intelligence' ? ' rel_ukey_all' : '');
    return (
        <div class={key_class}>
            <a href={kurl}>
                {k.name}
            </a>
        </div>
    )
}

const KeyList = props => {
    const lst = props.keys;
    const klst = lst.map((jkey, ix) => <Key jkey={jkey} />);
    const deleter = props.deleteKey;
    const inserter = props.insertKey;
    return (
        <div class="rel_parent_key">
            <div>
                <div class="rel_ukey_add" onClick={inserter}>+</div>
                <div class="rel_ukey_sub" onClick={deleter}>-</div>
            </div>
            <div id="keyList" class="rel_utags">
                {klst}
            </div>
        </div>
    )
}
class KeyComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = { keys: props.keys };
        this.deleteKey = this.deleteKey.bind(this);
        this.insertKey = this.insertKey.bind(this);
    }
    deleteKey() {
        const lst = this.state.keys;
        const klst = lst.map((jkey, ix) => jkey.name).filter(jkey => jkey !== "Artificial general intelligence");
        let keyname = prompt(`delete a keyword:\n${klst.join('\n')}`);
        fetch("/del_key/" + keyname)
            .then(response => {
                this.setState((prevState) => ({
                    keys: prevState.keys.filter(jkey => jkey.name !== keyname)
                }), () => {
                    console.log(response.text());
                });
            });
    }
    insertKey() {
        const lst = this.state.keys;
        const klst = lst.map((jkey, ix) => jkey.name);
        let keyname = prompt(`insert a keyword: `);
        fetch("/add_key/" + keyname)
            .then(response => {
                this.setState((prevState) => ({
                    keys: [...prevState.keys, { name: keyname, pids: [] }]
                }), () => {
                    console.log(response.text());
                });
            });
    }
    render() {
        return (
            <KeyList keys={this.state.keys} deleteKey={this.deleteKey} insertKey={this.insertKey} />
        );
    }
}




// render papers into #wrap
// ReactDOM.render(<PaperList papers={papers} tags={tags} />, document.getElementById('wrap'));
ReactDOM.render(<PaperList papers={papers} tags={tags} />, document.getElementById('wrap'));

// render tags into #tagwrap, if it exists
let tagwrap_elt = document.getElementById('tagwrap');
if (tagwrap_elt) {
    ReactDOM.render(<TagListComponent tags={tags} />, tagwrap_elt);
}

// render keys into #keywrap, if it exists
let keywrap_elt = document.getElementById('keywrap');
if (keywrap_elt) {
    ReactDOM.render(<KeyComponent keys={keys} />, keywrap_elt);
}

let tagcombwrap_elt = document.getElementById('tagcombwrap');
if (tagcombwrap_elt) {
    ReactDOM.render(<CombinedTagListComponent combined_tags={combined_tags} tags={tags} />, tagcombwrap_elt);
}

