NPM_PACK_OUT ?= npm.tgz
DIST_DIR ?= dist

.PHONY: default
default: $(NPM_PACK_OUT)

.PHONY: clean
clean:
	@rm -rf $(DIST_DIR) $(NPM_PACK_OUT)

$(DIST_DIR):
	@pnpm build

$(NPM_PACK_OUT): $(DIST_DIR)
	@pnpm pack --out $@
