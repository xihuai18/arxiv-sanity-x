from unittest.mock import patch


def test_tasks_paper_exists_treats_deleting_upload_as_missing():
    import tasks

    with patch("aslite.repositories.UploadedPaperRepository.get") as mget:
        mget.return_value = {"owner": "u", "parse_status": "ok", "deleting": True}
        assert tasks._paper_exists("up_abc") is False
