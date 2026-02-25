from chainlit.oauth_providers import providers, OAuthProvider
import logging

logger = logging.getLogger("chainlit")


def provider_already_registered(provider_instance: OAuthProvider) -> bool:
    return any(
        provider.id == provider_instance.id or type(provider) == type(provider_instance) for provider in providers
    )


def add_custom_oauth_provider(custom_provider_instance: OAuthProvider) -> None:
    if not custom_provider_instance.is_configured():
        logger.warning(
            f"Custom OAuth provider '{custom_provider_instance.id}' is not configured properly. Skipping registration."
        )
        return
    if provider_already_registered(custom_provider_instance):
        logger.warning(
            f"Custom OAuth provider '{custom_provider_instance.id}' is already registered. Skipping registration."
        )
        return
    providers.append(custom_provider_instance)
