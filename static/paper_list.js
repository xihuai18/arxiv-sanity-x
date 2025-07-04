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

// Multi-select dropdown component
const MultiSelectDropdown = props => {
    const {
        selectedTags,
        availableTags,
        isOpen,
        onToggle,
        onTagToggle,
        onRemoveTag,
        newTagValue,
        onNewTagChange,
        onAddNewTag,
        dropdownId,
        searchValue,
        onSearchChange,
        showNewTagInput = true
    } = props;

    const selectedTagElements = selectedTags.map((tag, ix) => (
        <div key={ix} class="multi-select-selected-tag">
            <span>{tag}</span>
            <span class="remove-tag" onClick={() => onRemoveTag(tag)}>×</span>
        </div>
    ));

    const triggerContent = selectedTags.length > 0 ? (
        <div class="multi-select-selected-tags">
            {selectedTagElements}
        </div>
    ) : (
        <div class="multi-select-placeholder">Select tags...</div>
    );

    // Filter tags by search value
    const filteredTags = availableTags.filter(tag =>
        tag.toLowerCase().includes(searchValue.toLowerCase())
    );

    const optionElements = filteredTags.map((tag, ix) => {
        const isSelected = selectedTags.includes(tag);
        return (
            <div key={ix} class="multi-select-option" onClick={() => onTagToggle(tag)}>
                <input
                    type="checkbox"
                    class="multi-select-checkbox"
                    checked={isSelected}
                    onChange={() => {}} // Handled by parent onClick
                />
                <span class="multi-select-option-text">{tag}</span>
            </div>
        );
    });

    return (
        <div class="multi-select-dropdown" id={dropdownId}>
            <div class="multi-select-trigger" onClick={onToggle}>
                <div class="multi-select-content">
                    {triggerContent}
                </div>
                <span class="multi-select-arrow">{isOpen ? '▲' : '▼'}</span>
            </div>
            {isOpen && (
                <div class="multi-select-dropdown-menu">
                    <div class="multi-select-search">
                        <input
                            type="text"
                            placeholder="Search tags..."
                            value={searchValue}
                            onChange={onSearchChange}
                            onClick={(e) => e.stopPropagation()}
                        />
                    </div>
                    {optionElements}
                    {showNewTagInput && (
                        <div class="multi-select-new-tag">
                            <input
                                type="text"
                                placeholder="Enter new tag..."
                                value={newTagValue}
                                onChange={onNewTagChange}
                                onKeyPress={(e) => e.key === 'Enter' && onAddNewTag()}
                                onClick={(e) => e.stopPropagation()}
                            />
                            <button
                                onClick={onAddNewTag}
                                disabled={!newTagValue.trim()}
                            >
                                Add
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

const Paper = props => {
    const p = props.paper;
    const lst = props.tags;
    const tlst = lst.map((jtag, ix) => jtag.name);
    const ulst = p.utags;

    const similar_url = "/?rank=pid&pid=" + encodeURIComponent(p.id);
    const inspect_url = "/inspect?pid=" + encodeURIComponent(p.id);
    const summary_url = "/summary?pid=" + encodeURIComponent(p.id);
    const thumb_img = p.thumb_url === '' ? null : <div class='rel_img'><img src={p.thumb_url} /></div>;

    // if the user is logged in then we can show the multi-select dropdown
    let utag_controls = null;
    if (user) {
        utag_controls = (
            <div class='rel_utags'>
                <MultiSelectDropdown
                    selectedTags={ulst}
                    availableTags={tlst}
                    isOpen={props.dropdownOpen}
                    onToggle={props.onToggleDropdown}
                    onTagToggle={props.onTagToggle}
                    onRemoveTag={props.onRemoveTag}
                    newTagValue={props.newTagValue}
                    onNewTagChange={props.onNewTagChange}
                    onAddNewTag={props.onAddNewTag}
                    dropdownId={props.dropdownId}
                    searchValue={props.searchValue}
                    onSearchChange={props.onSearchChange}
                />
            </div>
        )
    }

    return (
        <div class='rel_paper'>
            <div class="rel_score">
                {p.weight.toFixed(2)}
                {p.score_breakdown && (
                    <div class="score_breakdown" style={{fontSize: '0.8em', color: '#666', marginTop: '2px'}}>
                        {p.score_breakdown}
                    </div>
                )}
            </div>
            <div class='rel_title'><a href={'http://arxiv.org/abs/' + p.id} target="_blank" rel="noopener noreferrer">{p.title}</a></div>
            <div class='rel_authors'>{p.authors}</div>
            <div class="rel_time">{p.time}</div>
            <div class='rel_tags'>{p.tags}</div>
            {thumb_img}
            <div class='rel_abs'>{p.summary}</div>
            {utag_controls}
            <div class='paper-actions-footer'>
                <div class='rel_more'><a href={similar_url} target="_blank" rel="noopener noreferrer">Similar</a></div>
                <div class='rel_inspect'><a href={inspect_url} target="_blank" rel="noopener noreferrer">Inspect</a></div>
                <div class='rel_summary'><a href={summary_url} target="_blank" rel="noopener noreferrer">Summary</a></div>
                <div class='rel_alphaxiv'><a href={'https://www.alphaxiv.org/overview/' + p.id} target="_blank" rel="noopener noreferrer">alphaXiv</a></div>
                <div class='rel_cool'><a href={'https://papers.cool/arxiv/' + p.id} target="_blank" rel="noopener noreferrer">Cool</a></div>
            </div>
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
        this.state = {
            key: props.key,
            paper: props.paper,
            tags: props.tags,
            dropdownOpen: false,
            newTagValue: '',
            searchValue: ''
        };
        this.dropdownId = 'dropdown-' + props.paper.id;
        this.handleToggleDropdown = this.handleToggleDropdown.bind(this);
        this.handleTagToggle = this.handleTagToggle.bind(this);
        this.handleRemoveTag = this.handleRemoveTag.bind(this);
        this.handleNewTagChange = this.handleNewTagChange.bind(this);
        this.handleAddNewTag = this.handleAddNewTag.bind(this);
        this.handleSearchChange = this.handleSearchChange.bind(this);
        this.handleClickOutside = this.handleClickOutside.bind(this);
    }

    componentDidMount() {
        document.addEventListener('mousedown', this.handleClickOutside);
        this.handleKeyDown = this.handleKeyDown.bind(this);
        document.addEventListener('keydown', this.handleKeyDown);
    }

    componentWillUnmount() {
        document.removeEventListener('mousedown', this.handleClickOutside);
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    handleKeyDown(event) {
        if (event.key === 'Escape' && this.state.dropdownOpen) {
            this.setState({ dropdownOpen: false });
        }
    }

    handleClickOutside(event) {
        const dropdown = document.getElementById(this.dropdownId);
        if (dropdown && !dropdown.contains(event.target)) {
            this.setState({ dropdownOpen: false });
        }
    }

    handleToggleDropdown() {
        this.setState(prevState => ({
            dropdownOpen: !prevState.dropdownOpen,
            searchValue: !prevState.dropdownOpen ? '' : prevState.searchValue // Reset search when opened
        }));
    }

    handleSearchChange(event) {
        this.setState({
            searchValue: event.target.value
        });
    }

    handleTagToggle(tagName) {
        const { paper } = this.state;
        const isSelected = paper.utags.includes(tagName);

        if (isSelected) {
            // Remove tag
            fetch("/sub/" + paper.id + "/" + encodeURIComponent(tagName))
                .then(response => response.text())
                .then(text => {
                    if (text.startsWith('ok')) {
                        paper.utags = paper.utags.filter(tag => tag !== tagName);
                        this.setState({
                            paper: paper
                        });
                        console.log(`Removed tag: ${tagName}`);
                    } else {
                        console.error('Server error removing tag:', text);
                        alert('Failed to remove tag: ' + text);
                    }
                })
                .catch(error => {
                    console.error('Error removing tag:', error);
                    alert('Network error, failed to remove tag');
                });
        } else {
            // Add tag
            fetch("/add/" + paper.id + "/" + encodeURIComponent(tagName))
                .then(response => response.text())
                .then(text => {
                    if (text.startsWith('ok')) {
                        if (!paper.utags.includes(tagName)) {
                            paper.utags = [...paper.utags, tagName];
                        }
                        this.setState({
                            paper: paper
                        });
                        console.log(`Added tag: ${tagName}`);
                    } else {
                        console.error('Server error adding tag:', text);
                        alert('Failed to add tag: ' + text);
                    }
                })
                .catch(error => {
                    console.error('Error adding tag:', error);
                    alert('Network error, failed to add tag');
                });
        }
    }

    handleRemoveTag(tagName) {
        const { paper } = this.state;
        fetch("/sub/" + paper.id + "/" + encodeURIComponent(tagName))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    paper.utags = paper.utags.filter(tag => tag !== tagName);
                    this.setState({
                        paper: paper
                    });
                    console.log(`Removed tag: ${tagName}`);
                } else {
                    console.error('Server error removing tag:', text);
                    alert('Failed to remove tag: ' + text);
                }
            })
            .catch(error => {
                console.error('Error removing tag:', error);
                alert('Network error, failed to remove tag');
            });
    }

    handleNewTagChange(event) {
        this.setState({
            newTagValue: event.target.value
        });
    }

    handleAddNewTag() {
        const { paper, newTagValue } = this.state;
        const trimmedTag = newTagValue.trim();

        if (!trimmedTag) return;

        // Check if tag already exists
        if (paper.utags.includes(trimmedTag)) {
            alert('Tag already exists');
            return;
        }

        fetch("/add/" + paper.id + "/" + encodeURIComponent(trimmedTag))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    paper.utags = [...paper.utags, trimmedTag];
                    this.setState({
                        paper: paper,
                        newTagValue: ''
                    });
                    console.log(`Added new tag: ${trimmedTag}`);
                } else {
                    console.error('Server error adding new tag:', text);
                    alert('Failed to add new tag: ' + text);
                }
            })
            .catch(error => {
                console.error('Error adding new tag:', error);
                alert('Network error, failed to add new tag');
            });
    }

    render() {
        return (
            <Paper
                key={this.state.key}
                paper={this.state.paper}
                tags={this.state.tags}
                dropdownOpen={this.state.dropdownOpen}
                onToggleDropdown={this.handleToggleDropdown}
                onTagToggle={this.handleTagToggle}
                onRemoveTag={this.handleRemoveTag}
                newTagValue={this.state.newTagValue}
                onNewTagChange={this.handleNewTagChange}
                onAddNewTag={this.handleAddNewTag}
                dropdownId={this.dropdownId}
                searchValue={this.state.searchValue}
                onSearchChange={this.handleSearchChange}
            />
        );
    }
}

const Tag = props => {
    const t = props.tag;
    const turl = "/?rank=tags&tags=" + encodeURIComponent(t.name);
    const tag_class = 'rel_utag' + (t.name === 'all' ? ' rel_utag_all' : '');
    const isEditable = t.name !== 'all';

    return (
        <div class={tag_class + ' enhanced-tag'}>
            <a href={turl} class="tag-link">
                {t.n} {t.name}
            </a>
            {isEditable && (
                <div class="tag-actions">
                    <span class="tag-edit" onClick={() => props.onEdit(t)} title="Edit tag">✎</span>
                    <span class="tag-delete" onClick={() => props.onDelete(t)} title="Delete tag">×</span>
                </div>
            )}
        </div>
    )
}

const TagList = props => {
    const lst = props.tags;
    const tlst = lst.map((jtag, ix) =>
        <Tag
            key={ix}
            tag={jtag}
            onEdit={props.onEditTag}
            onDelete={props.onDeleteTag}
        />
    );

    // show the #wordwrap element if the user clicks inspect
    const show_inspect = () => {
        const wordwrap = document.getElementById("wordwrap");
        if (wordwrap.style.display === "block") {
            wordwrap.style.display = "none";
        } else {
            wordwrap.style.display = "block";
        }
    };
    const inspect_elt = words.length > 0 ? <div id="inspect_svm" onClick={show_inspect}>inspect</div> : null;

    return (
        <div class="enhanced-tag-list">
            <span class="tag-stats-inline">({lst.length} tags)</span>
            <br />
            <div id="tagList" class="rel_utags enhanced-tags">
                {tlst}
            </div>
            {inspect_elt}

            {/* Edit Modal */}
            {props.showEditModal && (
                <div class="modal-overlay" onClick={props.onCloseEditModal}>
                    <div class="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Edit Tag</h3>
                            <span class="modal-close" onClick={props.onCloseEditModal}>×</span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Tag Name:</label>
                                <input
                                    type="text"
                                    value={props.editingTagName}
                                    onChange={props.onEditingTagNameChange}
                                    class="form-input"
                                    placeholder="Enter new tag name"
                                />
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseEditModal}>Cancel</button>
                            <button class="btn btn-primary" onClick={props.onSaveTagEdit}>Save</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Add Tag Modal */}
            {props.showAddModal && (
                <div class="modal-overlay" onClick={props.onCloseAddModal}>
                    <div class="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Add Tag</h3>
                            <span class="modal-close" onClick={props.onCloseAddModal}>×</span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Tag Name:</label>
                                <input
                                    type="text"
                                    value={props.newTagName}
                                    onChange={props.onNewTagNameChange}
                                    class="form-input"
                                    placeholder="Enter tag name"
                                />
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseAddModal}>Cancel</button>
                            <button class="btn btn-primary" onClick={props.onSaveNewTag}>Save</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {props.showDeleteModal && (
                <div class="modal-overlay" onClick={props.onCloseDeleteModal}>
                    <div class="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Confirm Delete</h3>
                            <span class="modal-close" onClick={props.onCloseDeleteModal}>×</span>
                        </div>
                        <div class="modal-body">
                            <p>Are you sure you want to delete tag "<strong>{props.deletingTag && props.deletingTag.name}</strong>"?</p>
                            <p class="warning-text">This action is irreversible. All papers under this tag will lose the tag.</p>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseDeleteModal}>Cancel</button>
                            <button class="btn btn-danger" onClick={props.onConfirmDelete}>Delete</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}


class TagListComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            tags: props.tags,
            showEditModal: false,
            showDeleteModal: false,
            showAddModal: false,
            editingTag: null,
            editingTagName: '',
            deletingTag: null,
            newTagName: ''
        };
        this.handleEditTag = this.handleEditTag.bind(this);
        this.handleDeleteTag = this.handleDeleteTag.bind(this);
        this.handleAddTag = this.handleAddTag.bind(this);
        this.handleCloseEditModal = this.handleCloseEditModal.bind(this);
        this.handleCloseDeleteModal = this.handleCloseDeleteModal.bind(this);
        this.handleCloseAddModal = this.handleCloseAddModal.bind(this);
        this.handleEditingTagNameChange = this.handleEditingTagNameChange.bind(this);
        this.handleNewTagNameChange = this.handleNewTagNameChange.bind(this);
        this.handleSaveTagEdit = this.handleSaveTagEdit.bind(this);
        this.handleSaveNewTag = this.handleSaveNewTag.bind(this);
        this.handleConfirmDelete = this.handleConfirmDelete.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
    }

    componentDidMount() {
        document.addEventListener('keydown', this.handleKeyDown);
    }

    componentWillUnmount() {
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    handleKeyDown(event) {
        if (event.key === 'Escape') {
            if (this.state.showEditModal) {
                this.handleCloseEditModal();
            }
            if (this.state.showDeleteModal) {
                this.handleCloseDeleteModal();
            }
            if (this.state.showAddModal) {
                this.handleCloseAddModal();
            }
        }
    }

    handleEditTag(tag) {
        this.setState({
            showEditModal: true,
            editingTag: tag,
            editingTagName: tag.name
        });
    }

    handleDeleteTag(tag) {
        this.setState({
            showDeleteModal: true,
            deletingTag: tag
        });
    }

    handleAddTag() {
        this.setState({
            showAddModal: true,
            newTagName: ''
        });
    }

    handleCloseAddModal() {
        this.setState({
            showAddModal: false,
            newTagName: ''
        });
    }

    handleNewTagNameChange(event) {
        this.setState({ newTagName: event.target.value });
    }

    handleSaveNewTag() {
        const { newTagName } = this.state;
        if (!newTagName.trim()) {
            alert('Tag name cannot be empty');
            return;
        }

        // Should call Add tag API here
        console.log('Adding new tag:', newTagName.trim());
        // TODO: Implement Add tag logic

        this.setState({
            showAddModal: false,
            newTagName: ''
        });
    }


    handleCloseEditModal() {
        this.setState({
            showEditModal: false,
            editingTag: null,
            editingTagName: ''
        });
    }

    handleCloseDeleteModal() {
        this.setState({
            showDeleteModal: false,
            deletingTag: null
        });
    }

    handleEditingTagNameChange(event) {
        this.setState({ editingTagName: event.target.value });
    }

    handleSaveTagEdit() {
        const { editingTag, editingTagName } = this.state;
        if (!editingTagName.trim()) {
            alert('Tag name cannot be empty');
            return;
        }

        fetch("/rename/" + encodeURIComponent(editingTag.name) + "/" + encodeURIComponent(editingTagName.trim()))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState((prevState) => ({
                        tags: prevState.tags.map(tag =>
                            tag.name === editingTag.name
                                ? { ...tag, name: editingTagName.trim() }
                                : tag
                        ),
                        showEditModal: false,
                        editingTag: null,
                        editingTagName: ''
                    }));
                    console.log('Tag renamed successfully');
                } else {
                    alert('Rename failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error renaming tag:', error);
                alert('Network error, rename failed');
            });
    }

    handleConfirmDelete() {
        const { deletingTag } = this.state;

        fetch("/del/" + encodeURIComponent(deletingTag.name))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState((prevState) => ({
                        tags: prevState.tags.filter(tag => tag.name !== deletingTag.name),
                        showDeleteModal: false,
                        deletingTag: null
                    }));
                    console.log('Tag deleted successfully');
                } else {
                    alert('Delete failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error deleting tag:', error);
                alert('Network error, delete failed');
            });
    }

    render() {
        return (
            <TagList
                tags={this.state.tags}
                onEditTag={this.handleEditTag}
                onDeleteTag={this.handleDeleteTag}
                onAddTag={this.handleAddTag}
                showEditModal={this.state.showEditModal}
                showDeleteModal={this.state.showDeleteModal}
                showAddModal={this.state.showAddModal}
                editingTagName={this.state.editingTagName}
                newTagName={this.state.newTagName}
                deletingTag={this.state.deletingTag}
                onCloseEditModal={this.handleCloseEditModal}
                onCloseDeleteModal={this.handleCloseDeleteModal}
                onCloseAddModal={this.handleCloseAddModal}
                onEditingTagNameChange={this.handleEditingTagNameChange}
                onNewTagNameChange={this.handleNewTagNameChange}
                onSaveTagEdit={this.handleSaveTagEdit}
                onSaveNewTag={this.handleSaveNewTag}
                onConfirmDelete={this.handleConfirmDelete}
            />
        );
    }
}

const CombinedTag = props => {
    const t = props.comtag;
    const turl = "/?rank=tags&logic=and&tags=" + encodeURIComponent(t.name);
    const tag_class = 'rel_utag rel_utag_all enhanced-combined-tag';

    return (
        <div class={tag_class}>
            <a href={turl} class="combined-tag-link">
                {t.name}
            </a>
            <div class="combined-tag-actions">
                <span class="combined-tag-edit" onClick={() => props.onEdit(t)} title="Edit combined tag">✎</span>
                <span class="combined-tag-delete" onClick={() => props.onDelete(t)} title="Delete combined tag">×</span>
            </div>
        </div>
    )
}

const CombinedTagList = props => {
    const lst = props.combined_tags;
    const tlst = lst.map((jtag, ix) =>
        <CombinedTag
            key={ix}
            comtag={jtag}
            onEdit={props.onEditCombinedTag}
            onDelete={props.onDeleteCombinedTag}
        />
    );

    return (
        <div class="enhanced-combined-tag-list">
            <div class="combined-tag-list-actions">
                <span class="tag-stats-inline">({lst.length} combined tags)</span>
                <button class="tag-action-btn add-btn" onClick={props.onAddCombinedTag} title="Add new combined tag">
                    + Add
                </button>
            </div>
            <div id="combinedTagList" class="rel_utags enhanced-combined-tags">
                {tlst}
            </div>

            {/* Add/Edit Combined Tag Modal */}
            {props.showAddEditModal && (
                <div class="modal-overlay" onClick={props.onCloseAddEditModal}>
                    <div class="modal-content wide" onClick={(e) => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>{props.editingCombinedTag ? 'Edit Combined Tag' : 'Add Combined Tag'}</h3>
                            <span class="modal-close" onClick={props.onCloseAddEditModal}>×</span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Select tags to combine:</label>
                                <MultiSelectDropdown
                                    selectedTags={props.selectedTagsForCombination}
                                    availableTags={props.availableTagsForCombination}
                                    isOpen={props.combinationDropdownOpen}
                                    onToggle={props.onToggleCombinationDropdown}
                                    onTagToggle={props.onCombinationTagToggle}
                                    onRemoveTag={props.onRemoveCombinationTag}
                                    newTagValue=""
                                    onNewTagChange={() => {}}
                                    onAddNewTag={() => {}}
                                    dropdownId="combination-dropdown"
                                    searchValue={props.combinationSearchValue}
                                    onSearchChange={props.onCombinationSearchChange}
                                    showNewTagInput={false}
                                />
                            </div>
                            {props.selectedTagsForCombination.length > 0 && (
                                <div class="tag-combination-preview">
                                    <h4>Preview Combination:</h4>
                                    <div class="tag-combination-preview-tags">
                                        {props.selectedTagsForCombination.map((tag, ix) => (
                                            <span key={ix} class="tag-combination-preview-tag">{tag}</span>
                                        ))}
                                    </div>
                                    <p style={{ marginTop: '10px', fontSize: '12px', color: 'var(--text-color)', opacity: '0.8' }}>
                                        Combination Name: {props.selectedTagsForCombination.join(', ')}
                                    </p>
                                </div>
                            )}
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseAddEditModal}>Cancel</button>
                            <button
                                class="btn btn-primary"
                                onClick={props.onSaveCombinedTag}
                                disabled={props.selectedTagsForCombination.length < 2}
                            >
                                {props.editingCombinedTag ? 'Save' : 'Create'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {props.showDeleteModal && (
                <div class="modal-overlay" onClick={props.onCloseDeleteModal}>
                    <div class="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Confirm Delete</h3>
                            <span class="modal-close" onClick={props.onCloseDeleteModal}>×</span>
                        </div>
                        <div class="modal-body">
                            <p>Are you sure you want to delete combined tag "<strong>{props.deletingCombinedTag && props.deletingCombinedTag.name}</strong>"?</p>
                            <p class="warning-text">This action is irreversible.</p>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseDeleteModal}>Cancel</button>
                            <button class="btn btn-danger" onClick={props.onConfirmDelete}>Delete</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

class CombinedTagListComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            combined_tags: props.combined_tags,
            tags: props.tags,
            showAddEditModal: false,
            showDeleteModal: false,
            editingCombinedTag: null,
            deletingCombinedTag: null,
            selectedTagsForCombination: [],
            combinationDropdownOpen: false,
            combinationSearchValue: ''
        };
        this.handleAddCombinedTag = this.handleAddCombinedTag.bind(this);
        this.handleEditCombinedTag = this.handleEditCombinedTag.bind(this);
        this.handleDeleteCombinedTag = this.handleDeleteCombinedTag.bind(this);
        this.handleCloseAddEditModal = this.handleCloseAddEditModal.bind(this);
        this.handleCloseDeleteModal = this.handleCloseDeleteModal.bind(this);
        this.handleSaveCombinedTag = this.handleSaveCombinedTag.bind(this);
        this.handleConfirmDelete = this.handleConfirmDelete.bind(this);
        this.handleToggleCombinationDropdown = this.handleToggleCombinationDropdown.bind(this);
        this.handleCombinationTagToggle = this.handleCombinationTagToggle.bind(this);
        this.handleRemoveCombinationTag = this.handleRemoveCombinationTag.bind(this);
        this.handleCombinationSearchChange = this.handleCombinationSearchChange.bind(this);
        this.handleClickOutside = this.handleClickOutside.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
    }

    componentDidMount() {
        document.addEventListener('mousedown', this.handleClickOutside);
        document.addEventListener('keydown', this.handleKeyDown);
    }

    componentWillUnmount() {
        document.removeEventListener('mousedown', this.handleClickOutside);
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    handleKeyDown(event) {
        if (event.key === 'Escape') {
            if (this.state.showAddEditModal) {
                this.handleCloseAddEditModal();
            }
            if (this.state.showDeleteModal) {
                this.handleCloseDeleteModal();
            }
            if (this.state.combinationDropdownOpen) {
                this.setState({ combinationDropdownOpen: false });
            }
        }
    }

    handleClickOutside(event) {
        const dropdown = document.getElementById('combination-dropdown');
        if (dropdown && !dropdown.contains(event.target)) {
            this.setState({ combinationDropdownOpen: false });
        }
    }

    handleAddCombinedTag() {
        this.setState({
            showAddEditModal: true,
            editingCombinedTag: null,
            selectedTagsForCombination: [],
            combinationSearchValue: ''
        });
    }

    handleEditCombinedTag(combinedTag) {
        // Parse existing combined tags, compatible with "comma+space" or "comma only" formats
        const existingTags = combinedTag.name
            .split(',')
            .map(tag => tag.trim())
            .filter(tag => tag.length > 0);
        this.setState({
            showAddEditModal: true,
            editingCombinedTag: combinedTag,
            selectedTagsForCombination: existingTags,
            combinationSearchValue: ''
        });
    }

    handleDeleteCombinedTag(combinedTag) {
        this.setState({
            showDeleteModal: true,
            deletingCombinedTag: combinedTag
        });
    }


    handleCloseAddEditModal() {
        this.setState({
            showAddEditModal: false,
            editingCombinedTag: null,
            selectedTagsForCombination: [],
            combinationDropdownOpen: false,
            combinationSearchValue: ''
        });
    }

    handleCloseDeleteModal() {
        this.setState({
            showDeleteModal: false,
            deletingCombinedTag: null
        });
    }

    handleSaveCombinedTag() {
        const { editingCombinedTag, selectedTagsForCombination } = this.state;

        if (selectedTagsForCombination.length < 2) {
            alert('Please select at least two tags to combine');
            return;
        }

        const combinationName = selectedTagsForCombination.join(', ');

        if (editingCombinedTag) {
            // Edit existing combined tag: delete old one first, then add new one
            fetch("/del_ctag/" + encodeURIComponent(editingCombinedTag.name))
                .then(response => response.text())
                .then(text => {
                    if (text.includes('ok')) {
                        return fetch("/add_ctag/" + encodeURIComponent(combinationName));
                    } else {
                        throw new Error('Delete failed: ' + text);
                    }
                })
                .then(response => response.text())
                .then(text => {
                    if (text.includes('ok')) {
                        this.setState((prevState) => ({
                            combined_tags: prevState.combined_tags.map(tag =>
                                tag.name === editingCombinedTag.name
                                    ? { ...tag, name: combinationName }
                                    : tag
                            ),
                            showAddEditModal: false,
                            editingCombinedTag: null,
                            selectedTagsForCombination: [],
                            combinationDropdownOpen: false,
                            combinationSearchValue: ''
                        }));
                        console.log('Combined tag edited successfully');
                    } else {
                        throw new Error('Add failed: ' + text);
                    }
                })
                .catch(error => {
                    console.error('Error editing combined tag:', error);
                    alert('Edit failed: ' + error.message);
                });
        } else {
            // Add new combined tag
            fetch("/add_ctag/" + encodeURIComponent(combinationName))
                .then(response => response.text())
                .then(text => {
                    if (text.includes('ok')) {
                        this.setState((prevState) => ({
                            combined_tags: [...prevState.combined_tags, { name: combinationName }],
                            showAddEditModal: false,
                            selectedTagsForCombination: [],
                            combinationDropdownOpen: false,
                            combinationSearchValue: ''
                        }));
                        console.log('Combined tag added successfully');
                    } else {
                        alert('Add failed: ' + text);
                    }
                })
                .catch(error => {
                    console.error('Error adding combined tag:', error);
                    alert('Network error, add failed');
                });
        }
    }

    handleConfirmDelete() {
        const { deletingCombinedTag } = this.state;

        fetch("/del_ctag/" + encodeURIComponent(deletingCombinedTag.name))
            .then(response => response.text())
            .then(text => {
                if (text.includes('ok')) {
                    this.setState((prevState) => ({
                        combined_tags: prevState.combined_tags.filter(tag => tag.name !== deletingCombinedTag.name),
                        showDeleteModal: false,
                        deletingCombinedTag: null
                    }));
                    console.log('Combined tag deleted successfully');
                } else {
                    alert('Delete failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error deleting combined tag:', error);
                alert('Network error, delete failed');
            });
    }

    handleToggleCombinationDropdown() {
        this.setState(prevState => ({
            combinationDropdownOpen: !prevState.combinationDropdownOpen,
            combinationSearchValue: !prevState.combinationDropdownOpen ? '' : prevState.combinationSearchValue
        }));
    }

    handleCombinationTagToggle(tagName) {
        this.setState(prevState => {
            const isSelected = prevState.selectedTagsForCombination.includes(tagName);
            return {
                selectedTagsForCombination: isSelected
                    ? prevState.selectedTagsForCombination.filter(tag => tag !== tagName)
                    : [...prevState.selectedTagsForCombination, tagName]
            };
        });
    }

    handleRemoveCombinationTag(tagName) {
        this.setState(prevState => ({
            selectedTagsForCombination: prevState.selectedTagsForCombination.filter(tag => tag !== tagName)
        }));
    }

    handleCombinationSearchChange(event) {
        this.setState({
            combinationSearchValue: event.target.value
        });
    }

    render() {
        const availableTagsForCombination = this.state.tags
            .filter(tag => tag.name !== 'all')
            .map(tag => tag.name);

        return (
            <CombinedTagList
                combined_tags={this.state.combined_tags}
                searchValue={this.state.searchValue}
                onSearchChange={this.handleSearchChange}
                onAddCombinedTag={this.handleAddCombinedTag}
                onEditCombinedTag={this.handleEditCombinedTag}
                onDeleteCombinedTag={this.handleDeleteCombinedTag}
                showAddEditModal={this.state.showAddEditModal}
                showDeleteModal={this.state.showDeleteModal}
                editingCombinedTag={this.state.editingCombinedTag}
                deletingCombinedTag={this.state.deletingCombinedTag}
                onCloseAddEditModal={this.handleCloseAddEditModal}
                onCloseDeleteModal={this.handleCloseDeleteModal}
                onSaveCombinedTag={this.handleSaveCombinedTag}
                onConfirmDelete={this.handleConfirmDelete}
                selectedTagsForCombination={this.state.selectedTagsForCombination}
                availableTagsForCombination={availableTagsForCombination}
                combinationDropdownOpen={this.state.combinationDropdownOpen}
                onToggleCombinationDropdown={this.handleToggleCombinationDropdown}
                onCombinationTagToggle={this.handleCombinationTagToggle}
                onRemoveCombinationTag={this.handleRemoveCombinationTag}
                combinationSearchValue={this.state.combinationSearchValue}
                onCombinationSearchChange={this.handleCombinationSearchChange}
            />
        );
    }
}


const Key = props => {
    const k = props.jkey;
    const kurl = `/?q=${encodeURIComponent(k.name)}&rank=search`;
    const key_class = 'rel_ukey' + (k.name === 'Artificial general intelligence' ? ' rel_ukey_all' : '');
    const isEditable = k.name !== 'Artificial general intelligence';

    return (
        <div class={key_class + ' enhanced-keyword'}>
            <a href={kurl} class="keyword-link">
                {k.name}
            </a>
            {isEditable && (
                <div class="keyword-actions">
                    <span class="keyword-edit" onClick={() => props.onEdit(k)} title="Edit keyword">✎</span>
                    <span class="keyword-delete" onClick={() => props.onDelete(k)} title="Delete keyword">×</span>
                </div>
            )}
        </div>
    )
}

const KeyList = props => {
    const lst = props.keys;
    const klst = lst.map((jkey, ix) =>
        <Key
            key={ix}
            jkey={jkey}
            onEdit={props.onEditKey}
            onDelete={props.onDeleteKey}
        />
    );

    return (
        <div class="enhanced-keyword-list">
                <div class="keyword-list-actions">
                    <span class="tag-stats-inline">({lst.length} keywords)</span>
                    <button class="tag-action-btn add-btn" onClick={props.onAddKey} title="Add new keyword">
                        + Add
                    </button>
                </div>
            <div id="keyList" class="rel_utags enhanced-keywords">
                {klst}
            </div>

            {/* Add Keyword Modal */}
            {props.showAddModal && (
                <div class="modal-overlay" onClick={props.onCloseAddModal}>
                    <div class="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Add Keyword</h3>
                            <span class="modal-close" onClick={props.onCloseAddModal}>×</span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Keyword Name:</label>
                                <input
                                    type="text"
                                    value={props.newKeyName}
                                    onChange={props.onNewKeyNameChange}
                                    class="form-input"
                                    placeholder="Enter keyword name"
                                />
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseAddModal}>Cancel</button>
                            <button class="btn btn-primary" onClick={props.onSaveNewKey}>Save</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Edit Modal */}
            {props.showEditModal && (
                <div class="modal-overlay" onClick={props.onCloseEditModal}>
                    <div class="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Edit Keyword</h3>
                            <span class="modal-close" onClick={props.onCloseEditModal}>×</span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Keyword Name:</label>
                                <input
                                    type="text"
                                    value={props.editingKeyName}
                                    onChange={props.onEditingKeyNameChange}
                                    class="form-input"
                                    placeholder="Enter new keyword name"
                                />
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseEditModal}>Cancel</button>
                            <button class="btn btn-primary" onClick={props.onSaveKeyEdit}>Save</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {props.showDeleteModal && (
                <div class="modal-overlay" onClick={props.onCloseDeleteModal}>
                    <div class="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Confirm Delete</h3>
                            <span class="modal-close" onClick={props.onCloseDeleteModal}>×</span>
                        </div>
                        <div class="modal-body">
                            <p>Are you sure you want to delete keyword "<strong>{props.deletingKey && props.deletingKey.name}</strong>"?</p>
                            <p class="warning-text">This action is irreversible, all data related to this keyword will be deleted.</p>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseDeleteModal}>Cancel</button>
                            <button class="btn btn-danger" onClick={props.onConfirmDelete}>Delete</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

class KeyComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            keys: props.keys,
            showEditModal: false,
            showDeleteModal: false,
            showAddModal: false,
            editingKey: null,
            editingKeyName: '',
            deletingKey: null,
            newKeyName: ''
        };
        this.handleEditKey = this.handleEditKey.bind(this);
        this.handleDeleteKey = this.handleDeleteKey.bind(this);
        this.handleAddKey = this.handleAddKey.bind(this);
        this.handleCloseEditModal = this.handleCloseEditModal.bind(this);
        this.handleCloseDeleteModal = this.handleCloseDeleteModal.bind(this);
        this.handleCloseAddModal = this.handleCloseAddModal.bind(this);
        this.handleEditingKeyNameChange = this.handleEditingKeyNameChange.bind(this);
        this.handleNewKeyNameChange = this.handleNewKeyNameChange.bind(this);
        this.handleSaveKeyEdit = this.handleSaveKeyEdit.bind(this);
        this.handleSaveNewKey = this.handleSaveNewKey.bind(this);
        this.handleConfirmDelete = this.handleConfirmDelete.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
    }

    componentDidMount() {
        document.addEventListener('keydown', this.handleKeyDown);
    }

    componentWillUnmount() {
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    handleKeyDown(event) {
        if (event.key === 'Escape') {
            if (this.state.showEditModal) {
                this.handleCloseEditModal();
            }
            if (this.state.showDeleteModal) {
                this.handleCloseDeleteModal();
            }
            if (this.state.showAddModal) {
                this.handleCloseAddModal();
            }
        }
    }

    handleEditKey(key) {
        this.setState({
            showEditModal: true,
            editingKey: key,
            editingKeyName: key.name
        });
    }

    handleDeleteKey(key) {
        this.setState({
            showDeleteModal: true,
            deletingKey: key
        });
    }

    handleAddKey() {
        this.setState({
            showAddModal: true,
            newKeyName: ''
        });
    }

    handleCloseAddModal() {
        this.setState({
            showAddModal: false,
            newKeyName: ''
        });
    }

    handleNewKeyNameChange(event) {
        this.setState({ newKeyName: event.target.value });
    }

    handleSaveNewKey() {
        const { newKeyName } = this.state;
        if (!newKeyName.trim()) {
            alert('Keyword name cannot be empty');
            return;
        }

        const trimmedKey = newKeyName.trim();

        // Check if keyword already exists
        if (this.state.keys.some(key => key.name === trimmedKey)) {
            alert('Keyword already exists');
            return;
        }

        fetch("/add_key/" + encodeURIComponent(trimmedKey))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState((prevState) => ({
                        keys: [...prevState.keys, { name: trimmedKey, pids: [] }],
                        showAddModal: false,
                        newKeyName: ''
                    }));
                    console.log('Keyword added successfully');
                } else {
                    alert('Failed to add keyword: ' + text);
                }
            })
            .catch(error => {
                console.error('Error adding keyword:', error);
                alert('Network error, failed to add keyword');
            });
    }


    handleCloseEditModal() {
        this.setState({
            showEditModal: false,
            editingKey: null,
            editingKeyName: ''
        });
    }

    handleCloseDeleteModal() {
        this.setState({
            showDeleteModal: false,
            deletingKey: null
        });
    }

    handleEditingKeyNameChange(event) {
        this.setState({ editingKeyName: event.target.value });
    }

    handleSaveKeyEdit() {
        const { editingKey, editingKeyName } = this.state;
        if (!editingKeyName.trim()) {
            alert('Keyword name cannot be empty');
            return;
        }

        const trimmedKeyName = editingKeyName.trim();

        // Check if new name already exists
        if (this.state.keys.some(key => key.name === trimmedKeyName && key.name !== editingKey.name)) {
            alert('Keyword already exists');
            return;
        }

        // Should call keyword rename API here, but backend doesn't seem to have this interface
        // We delete the old one first, then add the new one
        fetch("/del_key/" + encodeURIComponent(editingKey.name))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    return fetch("/add_key/" + encodeURIComponent(trimmedKeyName));
                } else {
                    throw new Error('Delete failed: ' + text);
                }
            })
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState((prevState) => ({
                        keys: prevState.keys.map(key =>
                            key.name === editingKey.name
                                ? { ...key, name: trimmedKeyName }
                                : key
                        ),
                        showEditModal: false,
                        editingKey: null,
                        editingKeyName: ''
                    }));
                    console.log('Keyword renamed successfully');
                } else {
                    throw new Error('Add failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error renaming keyword:', error);
                alert('Rename failed: ' + error.message);
            });
    }

    handleConfirmDelete() {
        const { deletingKey } = this.state;

        fetch("/del_key/" + encodeURIComponent(deletingKey.name))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState((prevState) => ({
                        keys: prevState.keys.filter(key => key.name !== deletingKey.name),
                        showDeleteModal: false,
                        deletingKey: null
                    }));
                    console.log('Keyword deleted successfully');
                } else {
                    alert('Delete failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error deleting keyword:', error);
                alert('Network error, delete failed');
            });
    }

    render() {
        return (
            <KeyList
                keys={this.state.keys}
                onEditKey={this.handleEditKey}
                onDeleteKey={this.handleDeleteKey}
                onAddKey={this.handleAddKey}
                showEditModal={this.state.showEditModal}
                showDeleteModal={this.state.showDeleteModal}
                showAddModal={this.state.showAddModal}
                editingKeyName={this.state.editingKeyName}
                newKeyName={this.state.newKeyName}
                deletingKey={this.state.deletingKey}
                onCloseEditModal={this.handleCloseEditModal}
                onCloseDeleteModal={this.handleCloseDeleteModal}
                onCloseAddModal={this.handleCloseAddModal}
                onEditingKeyNameChange={this.handleEditingKeyNameChange}
                onNewKeyNameChange={this.handleNewKeyNameChange}
                onSaveKeyEdit={this.handleSaveKeyEdit}
                onSaveNewKey={this.handleSaveNewKey}
                onConfirmDelete={this.handleConfirmDelete}
            />
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
