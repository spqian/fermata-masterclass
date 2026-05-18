"""Regression test for the cloud-only auth crash on first sign-in.

In cloud (ADLS backend), reading a missing blob raises Azure SDK's
ResourceNotFoundError. Callers like upsert_oauth_user expect
FileNotFoundError (the stdlib type LocalObjectStorage raises) and use
that to fall through to "create on miss". Without translation, the
first sign-in for any new user blew up with HTTP 500.

These tests don't need a live Azure connection — they exercise the
static helper that does the type mapping.
"""
from __future__ import annotations


def test_translate_not_found_maps_azure_to_stdlib():
    from masterclass.storage.adls import AdlsObjectStorage
    from azure.core.exceptions import ResourceNotFoundError

    src = ResourceNotFoundError("the specified blob does not exist")
    translated = AdlsObjectStorage._translate_not_found(src, "tenant/foo/users/bar.json")
    assert isinstance(translated, FileNotFoundError)
    assert "tenant/foo/users/bar.json" in str(translated)


def test_translate_not_found_passes_through_other_errors():
    from masterclass.storage.adls import AdlsObjectStorage
    from azure.core.exceptions import HttpResponseError

    other = HttpResponseError("network blip")
    translated = AdlsObjectStorage._translate_not_found(other, "any-key")
    # Non-404 errors must propagate unchanged so the caller sees the real
    # cause (timeout, auth, permission, etc.) rather than a fake FileNotFound.
    assert translated is other
